"""
FastAPI application for exposing license analysis as a service
"""

import os
import sys
import logging
import asyncio
import tempfile
import io
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import existing modules
from core.github_utils import GitHubAPI, process_github_repository
from core.config import MAX_CONCURRENCY, RESULT_COLUMNS_ORDER
from core.utils import get_concluded_license, extract_thirdparty_dirs_column
from core.logging_utils import setup_logging
from core.email_utils import send_analysis_result, EmailConfig
from core.go_utils import get_github_url_from_pkggo
from core.maven_utils import analyze_maven_repository_url

import pandas as pd
from tqdm import tqdm

# Setup logging
loggers = setup_logging()
logger = loggers.get("main", logging.getLogger(__name__))


# ============================================================================
# Streaming Log Handler for API Responses
# ============================================================================
class AsyncStreamingLogHandler(logging.Handler):
    """自定义处理器，用于捕获特定级别的日志"""
    def __init__(self, log_queue: asyncio.Queue, min_level: int = logging.INFO):
        super().__init__()
        self.log_queue = log_queue
        self.min_level = min_level
    
    def emit(self, record: logging.LogRecord):
        """将日志消息发送到队列"""
        if record.levelno >= self.min_level:
            try:
                message = self.format(record)
                # 不阻塞，直接放入队列
                if not self.log_queue.full():
                    self.log_queue.put_nowait(message)
            except Exception:
                self.handleError(record)


class TqdmStreamingIO(io.StringIO):
    """用于捕获tqdm输出的自定义IO类"""
    def __init__(self, log_queue: asyncio.Queue):
        super().__init__()
        self.log_queue = log_queue

    def write(self, s):
        if s.strip():  # 只记录非空行
            try:
                if not self.log_queue.full():
                    self.log_queue.put_nowait(s.rstrip())
            except Exception:
                pass
        return super().write(s)


# ============================================================================
# Global SSE Broadcaster
# ============================================================================
MAX_SSE_SUBSCRIBERS = 10


class _Broadcaster:
    """Registry of per-client asyncio.Queues for SSE delivery."""

    def __init__(self):
        self._subscribers: list = []
        self._lock = asyncio.Lock()

    async def subscribe(self):
        async with self._lock:
            if len(self._subscribers) >= MAX_SSE_SUBSCRIBERS:
                return None
            q: asyncio.Queue = asyncio.Queue(maxsize=200)
            self._subscribers.append(q)
            return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def broadcast(self, message: str) -> None:
        """Non-blocking broadcast to all subscriber queues (drops if full)."""
        for q in self._subscribers:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass


broadcaster = _Broadcaster()


def _classify_log_type(msg: str) -> str:
    """Mirror the classification logic in stream-parser.js."""
    if "[PROGRESS]" in msg or "已完成" in msg:
        return "progress"
    if "[SUCCESS]" in msg or "成功" in msg or "完成" in msg:
        return "success"
    if "[ERROR]" in msg or "ERROR" in msg or "失败" in msg or "错误" in msg:
        return "error"
    if "[WARNING]" in msg or "WARNING" in msg or "警告" in msg:
        return "warning"
    if "[START]" in msg or "开始处理" in msg or "开始分析" in msg:
        return "start"
    return "info"


class GlobalBroadcastLogHandler(logging.Handler):
    """Pushes every INFO+ log record to the global SSE broadcaster as JSON."""

    def __init__(self, job_id: str, min_level: int = logging.INFO):
        super().__init__()
        self.job_id = job_id
        self.min_level = min_level

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < self.min_level:
            return
        try:
            raw = self.format(record)
            msg_type = _classify_log_type(raw)
            payload = json.dumps({
                "job_id": self.job_id,
                "type": msg_type,
                "message": raw,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, ensure_ascii=False)
            broadcaster.broadcast(payload)
        except Exception:
            self.handleError(record)


# Create FastAPI app
app = FastAPI(
    title="GitHub License Analyzer API",
    description="API for analyzing GitHub repository licenses",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global API instance
_api_instance = None


async def get_github_api():
    """Get or create GitHub API instance"""
    global _api_instance
    if _api_instance is None:
        _api_instance = GitHubAPI()
        await _api_instance.initialize()
    return _api_instance


@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    logger.info("API启动中...")
    try:
        api = await get_github_api()
        logger.info("GitHub API客户端初始化成功")
    except Exception as e:
        logger.error(f"API初始化失败: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("API关闭中...")
    global _api_instance
    if _api_instance:
        await _api_instance.close()


@app.get("/")
async def root():
    """Redirect to frontend"""
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "GitHub License Analyzer API"
    }


@app.get("/api/v1/logs/live")
async def live_logs():
    """SSE stream of all API activity for the frontend monitor panel."""
    q = await broadcaster.subscribe()
    if q is None:
        raise HTTPException(status_code=503, detail="Too many monitor connections (max 10)")

    async def event_stream() -> AsyncGenerator[str, None]:
        yield "data: " + json.dumps({
            "job_id": "system",
            "type": "info",
            "message": "已连接到服务器日志流",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }, ensure_ascii=False) + "\n\n"

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await broadcaster.unsubscribe(q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.post("/api/v1/analyze")
async def analyze_licenses(
    file: UploadFile = File(...),
    email: str = Form(...),
    smtp_server: Optional[str] = Form(None),
    smtp_port: Optional[int] = Form(None),
):
    """
    分析许可证信息并发送结果到邮箱
    
    Args:
        file: 上传的Excel文件 (input.xlsx)
        email: 接收结果的邮箱地址
        smtp_server: SMTP服务器地址（可选，使用环境变量）
        smtp_port: SMTP端口（可选，使用环境变量）
        
    Returns:
        分析结果和邮件发送状态
    """
    temp_input = None
    temp_output = None
    job_id = str(uuid.uuid4())[:8]
    global_handler = GlobalBroadcastLogHandler(job_id=job_id)
    global_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    _main_logger = logging.getLogger('__main__')
    _core_logger = logging.getLogger('main')
    _main_logger.addHandler(global_handler)
    _core_logger.addHandler(global_handler)

    try:
        broadcaster.broadcast(json.dumps({
            "job_id": job_id, "type": "start",
            "message": f"[START] 作业 {job_id}: 开始处理文件 {file.filename}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }, ensure_ascii=False))

        # 验证邮箱格式
        if not email or "@" not in email:
            raise HTTPException(status_code=400, detail="无效的邮箱地址")

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_input:
            temp_input = tmp_input.name
            content = await file.read()
            tmp_input.write(content)
            tmp_input.flush()

        logger.info(f"接收到文件: {file.filename}, 保存至: {temp_input}")

        # 读取Excel文件
        try:
            df = pd.read_excel(temp_input)
            logger.info(f"读取了 {len(df)} 行数据")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"无法读取Excel文件: {str(e)}"
            )

        # 初始化API
        api = await get_github_api()

        # 处理仓库
        logger.info("开始处理仓库...")
        results = await _process_repositories(api, df)

        # 生成输出
        logger.info("生成输出文件...")
        output_df = _generate_output(results)

        # 保存输出文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_output:
            temp_output = tmp_output.name

        with pd.ExcelWriter(temp_output, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='分析结果', index=False)

        logger.info(f"输出文件已保存: {temp_output}")

        # 发送邮件
        logger.info(f"发送结果到邮箱: {email}")
        email_config = None
        if smtp_server or smtp_port:
            email_config = EmailConfig(
                smtp_server=smtp_server,
                smtp_port=smtp_port
            )

        email_sent = send_analysis_result(
            recipient_email=email,
            output_file_path=temp_output,
            smtp_config=email_config
        )

        # 返回结果
        return {
            "status": "success",
            "message": "分析完成" + ("，结果已发送至邮箱" if email_sent else "，但邮件发送失败"),
            "processed_rows": len(results),
            "email_sent": email_sent,
            "email": email,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {str(e)}"
        )
    finally:
        _main_logger.removeHandler(global_handler)
        _core_logger.removeHandler(global_handler)
        broadcaster.broadcast(json.dumps({
            "job_id": job_id, "type": "success",
            "message": f"[SUCCESS] 作业 {job_id}: 处理完成",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }, ensure_ascii=False))
        # 清理临时文件
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
                logger.debug(f"已删除临时输入文件: {temp_input}")
            except Exception as e:
                logger.warning(f"删除临时输入文件失败: {e}")

        if temp_output and os.path.exists(temp_output):
            try:
                os.unlink(temp_output)
                logger.debug(f"已删除临时输出文件: {temp_output}")
            except Exception as e:
                logger.warning(f"删除临时输出文件失败: {e}")


@app.post("/api/v1/analyze-and-download")
async def analyze_and_download(
    file: UploadFile = File(...)
):
    """
    分析许可证信息并返回Excel文件（不发送邮件）
    
    Args:
        file: 上传的Excel文件 (input.xlsx)
        
    Returns:
        输出Excel文件
    """
    temp_input = None
    temp_output = None
    job_id = str(uuid.uuid4())[:8]
    global_handler = GlobalBroadcastLogHandler(job_id=job_id)
    global_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    _main_logger = logging.getLogger('__main__')
    _core_logger = logging.getLogger('main')
    _main_logger.addHandler(global_handler)
    _core_logger.addHandler(global_handler)

    try:
        broadcaster.broadcast(json.dumps({
            "job_id": job_id, "type": "start",
            "message": f"[START] 作业 {job_id}: 开始处理文件 {file.filename}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }, ensure_ascii=False))

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_input:
            temp_input = tmp_input.name
            content = await file.read()
            tmp_input.write(content)
            tmp_input.flush()

        logger.info(f"接收到文件: {file.filename}, 保存至: {temp_input}")

        # 读取Excel文件
        try:
            df = pd.read_excel(temp_input)
            logger.info(f"读取了 {len(df)} 行数据")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"无法读取Excel文件: {str(e)}"
            )

        # 初始化API
        api = await get_github_api()

        # 处理仓库
        logger.info("开始处理仓库...")
        results = await _process_repositories(api, df)

        # 生成输出
        logger.info("生成输出文件...")
        output_df = _generate_output(results)

        # 保存输出文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_output:
            temp_output = tmp_output.name

        with pd.ExcelWriter(temp_output, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='分析结果', index=False)

        logger.info(f"输出文件已保存: {temp_output}")

        # 返回文件
        return FileResponse(
            path=temp_output,
            filename=f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {str(e)}"
        )
    finally:
        _main_logger.removeHandler(global_handler)
        _core_logger.removeHandler(global_handler)
        broadcaster.broadcast(json.dumps({
            "job_id": job_id, "type": "success",
            "message": f"[SUCCESS] 作业 {job_id}: 处理完成",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }, ensure_ascii=False))
        # 注意：FileResponse会自动清理文件，所以这里只清理输入文件
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
                logger.debug(f"已删除临时输入文件: {temp_input}")
            except Exception as e:
                logger.warning(f"删除临时输入文件失败: {e}")


@app.post("/api/v1/analyze-stream")
async def analyze_with_stream(
    file: UploadFile = File(...)
):
    """
    分析许可证信息并流式返回日志（错误、警告和进度条）
    
    Args:
        file: 上传的Excel文件 (input.xlsx)
        
    Returns:
        流式日志响应
    """
    async def log_generator() -> AsyncGenerator[str, None]:
        temp_input = None
        job_id = str(uuid.uuid4())[:8]

        try:
            # 创建日志队列（容纳更多日志消息）
            log_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_input:
                temp_input = tmp_input.name
                content = await file.read()
                tmp_input.write(content)
                tmp_input.flush()

            # 添加流式日志处理器
            stream_handler = AsyncStreamingLogHandler(
                log_queue=log_queue,
                min_level=logging.INFO
            )
            stream_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            global_handler = GlobalBroadcastLogHandler(job_id=job_id)
            global_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )

            # 为所有相关日志记录器添加处理器
            main_logger = logging.getLogger('__main__')
            main_logger.addHandler(stream_handler)
            main_logger.addHandler(global_handler)

            # 也添加到实际使用的日志记录器
            core_logger = logging.getLogger('main')
            core_logger.addHandler(stream_handler)
            core_logger.addHandler(global_handler)

            try:
                broadcaster.broadcast(json.dumps({
                    "job_id": job_id, "type": "start",
                    "message": f"[START] 作业 {job_id}: 开始处理文件 {file.filename}",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, ensure_ascii=False))
                yield f"[START] 开始处理文件: {file.filename}\n"

                # 读取Excel文件
                try:
                    df = pd.read_excel(temp_input)
                    yield f"[INFO] 读取了 {len(df)} 行数据\n"
                except Exception as e:
                    yield f"[ERROR] 无法读取Excel文件: {str(e)}\n"
                    return

                # 初始化API
                api = await get_github_api()
                yield "[INFO] GitHub API 客户端已初始化\n"

                # 处理仓库（带流式日志）
                yield "[INFO] 开始处理仓库...\n"

                # 创建一个任务来处理仓库
                process_task = asyncio.create_task(
                    _process_repositories(api, df, log_queue)
                )

                # 实时读取日志队列
                while not process_task.done():
                    try:
                        # 尝试从队列中获取日志（非阻塞）
                        try:
                            log_msg = log_queue.get_nowait()
                            yield f"{log_msg}\n"
                        except asyncio.QueueEmpty:
                            # 队列为空，等待一下再继续
                            await asyncio.sleep(0.05)
                    except Exception as e:
                        yield f"[ERROR] 日志处理错误: {str(e)}\n"
                        break

                # 等待任务完成
                results = await process_task

                # 输出所有剩余的队列消息
                while True:
                    try:
                        log_msg = log_queue.get_nowait()
                        yield f"{log_msg}\n"
                    except asyncio.QueueEmpty:
                        break

                broadcaster.broadcast(json.dumps({
                    "job_id": job_id, "type": "success",
                    "message": f"[SUCCESS] 作业 {job_id}: 处理完成，共处理 {len(results)} 行数据",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, ensure_ascii=False))
                yield f"[SUCCESS] 处理完成，共处理 {len(results)} 行数据\n"

            finally:
                # 移除处理器
                main_logger.removeHandler(stream_handler)
                main_logger.removeHandler(global_handler)
                core_logger.removeHandler(global_handler)

        except Exception as e:
            yield f"[ERROR] 处理请求时出错: {str(e)}\n"
        finally:
            # 清理临时文件
            if temp_input and os.path.exists(temp_input):
                try:
                    os.unlink(temp_input)
                except Exception:
                    pass

    return StreamingResponse(
        log_generator(),
        media_type="text/plain; charset=utf-8"
    )


@app.post("/api/v1/analyze-stream-email")
async def analyze_with_stream_and_email(
    file: UploadFile = File(...),
    email: str = Form(...),
    smtp_server: Optional[str] = Form(None),
    smtp_port: Optional[int] = Form(None),
):
    """
    分析许可证信息，流式返回日志，完成后发送邮件
    
    Args:
        file: 上传的Excel文件 (input.xlsx)
        email: 接收结果的邮箱地址
        smtp_server: SMTP服务器地址（可选）
        smtp_port: SMTP端口（可选）
        
    Returns:
        流式日志响应，最后输出邮件发送状态
    """
    async def log_generator() -> AsyncGenerator[str, None]:
        temp_input = None
        temp_output = None
        job_id = str(uuid.uuid4())[:8]

        try:
            # 验证邮箱格式
            if not email or "@" not in email:
                yield "[ERROR] 无效的邮箱地址\n"
                return

            # 创建日志队列（容纳更多日志消息）
            log_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_input:
                temp_input = tmp_input.name
                content = await file.read()
                tmp_input.write(content)
                tmp_input.flush()

            # 添加流式日志处理器
            stream_handler = AsyncStreamingLogHandler(
                log_queue=log_queue,
                min_level=logging.INFO
            )
            stream_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            global_handler = GlobalBroadcastLogHandler(job_id=job_id)
            global_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )

            # 为所有相关日志记录器添加处理器
            main_logger = logging.getLogger('__main__')
            main_logger.addHandler(stream_handler)
            main_logger.addHandler(global_handler)

            # 也添加到实际使用的日志记录器
            core_logger = logging.getLogger('main')
            core_logger.addHandler(stream_handler)
            core_logger.addHandler(global_handler)

            try:
                broadcaster.broadcast(json.dumps({
                    "job_id": job_id, "type": "start",
                    "message": f"[START] 作业 {job_id}: 开始处理文件 {file.filename}",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, ensure_ascii=False))
                yield f"[START] 开始处理文件: {file.filename}\n"

                # 读取Excel文件
                try:
                    df = pd.read_excel(temp_input)
                    yield f"[INFO] 读取了 {len(df)} 行数据\n"
                except Exception as e:
                    yield f"[ERROR] 无法读取Excel文件: {str(e)}\n"
                    return

                # 初始化API
                api = await get_github_api()
                yield "[INFO] GitHub API 客户端已初始化\n"

                # 处理仓库（带流式日志）
                yield "[INFO] 开始处理仓库...\n"

                # 创建一个任务来处理仓库
                process_task = asyncio.create_task(
                    _process_repositories(api, df, log_queue)
                )

                # 实时读取日志队列
                while not process_task.done():
                    try:
                        try:
                            log_msg = log_queue.get_nowait()
                            yield f"{log_msg}\n"
                        except asyncio.QueueEmpty:
                            await asyncio.sleep(0.05)
                    except Exception as e:
                        yield f"[ERROR] 日志处理错误: {str(e)}\n"
                        break

                # 等待任务完成
                results = await process_task

                # 输出所有剩余的队列消息
                while True:
                    try:
                        log_msg = log_queue.get_nowait()
                        yield f"{log_msg}\n"
                    except asyncio.QueueEmpty:
                        break

                yield f"[INFO] 处理完成，共处理 {len(results)} 行数据\n"

                # 生成输出
                yield "[INFO] 生成输出文件...\n"
                output_df = _generate_output(results)

                # 保存输出文件
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_output:
                    temp_output = tmp_output.name

                with pd.ExcelWriter(temp_output, engine='openpyxl') as writer:
                    output_df.to_excel(writer, sheet_name='分析结果', index=False)

                yield "[INFO] 输出文件已生成\n"

                # 发送邮件
                yield f"[INFO] 正在发送结果到邮箱: {email}\n"
                email_config = None
                if smtp_server or smtp_port:
                    email_config = EmailConfig(
                        smtp_server=smtp_server,
                        smtp_port=smtp_port
                    )

                email_sent = send_analysis_result(
                    recipient_email=email,
                    output_file_path=temp_output,
                    smtp_config=email_config
                )

                if email_sent:
                    broadcaster.broadcast(json.dumps({
                        "job_id": job_id, "type": "success",
                        "message": f"[SUCCESS] 作业 {job_id}: 邮件已发送到 {email}",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }, ensure_ascii=False))
                    yield f"[SUCCESS] 邮件已发送到: {email}\n"
                else:
                    yield f"[ERROR] 邮件发送失败，请检查配置\n"

            finally:
                # 移除处理器
                main_logger.removeHandler(stream_handler)
                main_logger.removeHandler(global_handler)
                core_logger.removeHandler(global_handler)

        except Exception as e:
            yield f"[ERROR] 处理请求时出错: {str(e)}\n"
        finally:
            # 清理临时文件
            if temp_input and os.path.exists(temp_input):
                try:
                    os.unlink(temp_input)
                except Exception:
                    pass
            if temp_output and os.path.exists(temp_output):
                try:
                    os.unlink(temp_output)
                except Exception:
                    pass
    
    return StreamingResponse(
        log_generator(),
        media_type="text/plain; charset=utf-8"
    )


@app.post("/api/v1/analyze-stream-download")
async def analyze_with_stream_and_download(
    file: UploadFile = File(...)
):
    """
    分析许可证信息，流式返回日志，完成后返回Excel文件
    
    Args:
        file: 上传的Excel文件 (input.xlsx)
        
    Returns:
        先流式输出日志，然后返回Excel文件
    """
    temp_input = None
    temp_output = None
    
    try:
        # 创建日志队列（容纳更多日志消息）
        log_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_input:
            temp_input = tmp_input.name
            content = await file.read()
            tmp_input.write(content)
            tmp_input.flush()
        
        # 添加流式日志处理器
        stream_handler = AsyncStreamingLogHandler(
            log_queue=log_queue,
            min_level=logging.INFO
        )
        stream_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # 为所有相关日志记录器添加处理器
        main_logger = logging.getLogger('__main__')
        main_logger.addHandler(stream_handler)
        
        # 也添加到实际使用的日志记录器
        core_logger = logging.getLogger('main')
        core_logger.addHandler(stream_handler)
        
        try:
            logger.info(f"接收到文件: {file.filename}, 保存至: {temp_input}")
            
            # 读取Excel文件
            try:
                df = pd.read_excel(temp_input)
                logger.info(f"读取了 {len(df)} 行数据")
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"无法读取Excel文件: {str(e)}"
                )
            
            # 初始化API
            api = await get_github_api()
            
            # 处理仓库
            logger.info("开始处理仓库...")
            results = await _process_repositories(api, df, log_queue)
            
            # 生成输出
            logger.info("生成输出文件...")
            output_df = _generate_output(results)
            
            # 保存输出文件
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_output:
                temp_output = tmp_output.name
            
            with pd.ExcelWriter(temp_output, engine='openpyxl') as writer:
                output_df.to_excel(writer, sheet_name='分析结果', index=False)
            
            logger.info(f"输出文件已保存: {temp_output}")
            
            # 返回文件
            return FileResponse(
                path=temp_output,
                filename=f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
        finally:
            # 移除处理器
            main_logger.removeHandler(stream_handler)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
                logger.debug(f"已删除临时输入文件: {temp_input}")
            except Exception as e:
                logger.warning(f"删除临时输入文件失败: {e}")



async def _process_repositories(api, df, log_queue=None):
    """Process all repositories with concurrency control"""
    from core.github_utils import normalize_github_url
    import re
    
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    results = {}
    completed_count = 0
    total_count = len(df)
    lock = asyncio.Lock()
    
    async def log_progress():
        """输出处理进度"""
        progress_pct = (completed_count / total_count * 100) if total_count > 0 else 0
        msg = f"[PROGRESS] 已完成 {completed_count}/{total_count} ({progress_pct:.1f}%)"
        if log_queue:
            try:
                log_queue.put_nowait(msg)
            except:
                pass
        logger.info(msg)
    
    async def process_single(row, index):
        nonlocal completed_count
        async with sem:
            name = None
            try:
                original_url = row.get("github_url")
                url = normalize_github_url(original_url)
                version = row.get("version")
                name = row.get("name", None)
                
                # Check if it's a Go package
                is_go_pkg = False
                if isinstance(url, str):
                    if url.startswith("https://pkg.go.dev/") or url.startswith("https://go.dev/") or re.match(r"^go\.[\w\.-]+/", url):
                        is_go_pkg = True
                
                if is_go_pkg:
                    logger.info(f"检测到 Go 包 URL: {url}")
                    if log_queue:
                        try:
                            log_queue.put_nowait(f"[INFO] 检测到 Go 包 URL: {url}")
                        except:
                            pass
                    github_info = await get_github_url_from_pkggo(url, version, name)
                    github_url = github_info.get("github_url")
                    if github_url:
                        result = await process_github_repository(api, github_url, version, name=name)
                    else:
                        result = await process_github_repository(api, url, version, name=name)
                else:
                    # Check if it's a Maven URL
                    is_maven_url = isinstance(url, str) and (
                        "mvnrepository.com/artifact" in url or 
                        "repo1.maven.org/maven2" in url
                    )
                    if is_maven_url:
                        logger.info(f"检测到 Maven URL: {url}")
                        if log_queue:
                            try:
                                log_queue.put_nowait(f"[INFO] 检测到 Maven URL: {url}")
                            except:
                                pass
                        result = await process_github_repository(api, url, version, name=name)
                        
                        if result.get("status") != "success":
                            logger.info(f"GitHub 流程未成功，调用 Maven 处理函数")
                            if log_queue:
                                try:
                                    log_queue.put_nowait(f"[INFO] GitHub 流程未成功，调用 Maven 处理函数")
                                except:
                                    pass
                            try:
                                maven_result = analyze_maven_repository_url(url)
                                license_file_url = f"https://mvnrepository.com/artifact/{maven_result['group_id']}/{maven_result['artifact_id']}/{maven_result.get('version', '')}"
                                
                                copyright_notice = maven_result.get('copyright')
                                if not copyright_notice:
                                    org_parts = maven_result['group_id'].split(".")
                                    orgname = org_parts[1] if len(org_parts) > 1 else org_parts[0]
                                    copyright_notice = f"Copyright (c) {datetime.now().year} {orgname.capitalize()}"
                                
                                result = {
                                    "input_url": original_url,
                                    "repo_url": None,
                                    "input_version": version,
                                    "resolved_version": maven_result.get('version'),
                                    "used_default_branch": False,
                                    "component_name": name or maven_result['artifact_id'],
                                    "license_files": license_file_url,
                                    "license_analysis": {
                                        "license_determination_reason": "Fetched from Maven Central POM",
                                        "license_source": maven_result.get('license_source', 'maven_central')
                                    },
                                    "license_type": maven_result.get('license'),
                                    "has_license_conflict": False,
                                    "readme_license": None,
                                    "license_file_license": maven_result.get('license'),
                                    "copyright_notice": copyright_notice,
                                    "status": "success",
                                    "input_name": name,
                                }
                            except Exception as e:
                                logger.warning(f"Maven 处理失败: {e}")
                                if log_queue:
                                    try:
                                        log_queue.put_nowait(f"[WARNING] Maven 处理失败: {e}")
                                    except:
                                        pass
                    else:
                        result = await process_github_repository(api, url, version, name=name)
                
                result["input_name"] = name
                result["input_url"] = original_url
                results[index] = result
                
            except Exception as e:
                logger.error(f"处理失败 {row.get('github_url')}: {e}", exc_info=True)
                if log_queue:
                    try:
                        log_queue.put_nowait(f"[ERROR] 处理失败 {row.get('github_url')}: {e}")
                    except:
                        pass
                results[index] = {
                    "input_url": row.get("github_url"),
                    "status": "error",
                    "error": str(e),
                    "input_name": name
                }
            finally:
                # 更新进度计数
                async with lock:
                    completed_count += 1
                    # 每完成5个任务或最后一个任务时输出进度
                    if completed_count % 5 == 0 or completed_count == total_count:
                        await log_progress()
    
    # Process all rows
    tasks = [process_single(row, idx) for idx, row in df.iterrows()]
    await asyncio.gather(*tasks)
    
    # Return ordered results
    return [results[i] for i in range(len(df)) if i in results]


def _generate_output(results):
    """Generate output DataFrame from results"""
    output_df = pd.DataFrame(results)
    
    # Add concluded_license
    output_df['concluded_license'] = output_df.apply(
        lambda row: get_concluded_license(
            row.get('license_type'),
            row.get('readme_license'),
            row.get('license_file_license')
        ),
        axis=1
    )
    
    # Extract thirdparty_dirs
    output_df = extract_thirdparty_dirs_column(output_df)
    
    # Append " AND Others" if thirdparty_dirs exist
    def _has_thirdparty(row):
        analysis = row.get("license_analysis")
        if isinstance(analysis, dict):
            dirs = analysis.get("thirdparty_dirs")
            if isinstance(dirs, list) and len(dirs) > 0:
                return True
        tp_col = row.get("thirdparty_dirs")
        return isinstance(tp_col, str) and tp_col.strip() != ""
    
    def _append_others(expr: str) -> str:
        expr = (expr or "").strip()
        if not expr:
            return "Others"
        if not expr.endswith(" AND Others"):
            return f"{expr} AND Others"
        return expr
    
    output_df["concluded_license"] = output_df.apply(
        lambda r: _append_others(r.get("concluded_license"))
        if _has_thirdparty(r) else r.get("concluded_license"),
        axis=1
    )
    
    # Reorder columns
    existing_columns = [col for col in RESULT_COLUMNS_ORDER if col in output_df.columns]
    remaining_columns = [col for col in output_df.columns if col not in existing_columns]
    final_columns = existing_columns + remaining_columns
    output_df = output_df[final_columns]
    
    return output_df


# Mount static files (must be after all route definitions)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    import uvicorn
    
    logger.info(f"启动API服务器: {host}:{port}")
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    run_api_server()
