"""
Email utility functions for sending analysis results
"""

import os
import logging
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from email.encoders import encode_base64
from typing import List, Optional


logger = logging.getLogger(__name__)


class EmailConfig:
    """邮件配置类"""
    
    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = None,
        sender_email: str = None,
        sender_password: str = None,
        use_tls: bool = True
    ):
        # 从环境变量或参数读取配置
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL")
        self.sender_password = sender_password or os.getenv("SENDER_PASSWORD")
        self.use_tls = use_tls
        
        # 验证必要的配置
        if not self.sender_email:
            raise ValueError("SENDER_EMAIL environment variable not set")
        if not self.sender_password:
            raise ValueError("SENDER_PASSWORD environment variable not set")


class EmailSender:
    """邮件发送器"""
    
    def __init__(self, config: EmailConfig = None):
        self.config = config or EmailConfig()
    
    def send_email(
        self,
        recipient_email: str,
        subject: str,
        body: str,
        attachment_path: Optional[str] = None,
        attachment_name: Optional[str] = None,
        cc_list: Optional[List[str]] = None,
        bcc_list: Optional[List[str]] = None
    ) -> bool:
        """
        发送邮件
        
        Args:
            recipient_email: 收件人邮箱
            subject: 邮件主题
            body: 邮件正文
            attachment_path: 附件路径
            attachment_name: 附件显示名称
            cc_list: 抄送列表
            bcc_list: 密送列表
            
        Returns:
            成功返回True，失败返回False
        """
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.config.sender_email
            msg['To'] = recipient_email
            msg['Date'] = formatdate(localtime=True)
            msg['Subject'] = subject
            
            # 添加抄送和密送
            if cc_list:
                msg['Cc'] = ', '.join(cc_list)
            if bcc_list:
                msg['Bcc'] = ', '.join(bcc_list)
            
            # 添加邮件正文
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 添加附件
            if attachment_path and os.path.exists(attachment_path):
                self._attach_file(msg, attachment_path, attachment_name)
            
            # 发送邮件
            all_recipients = [recipient_email]
            if cc_list:
                all_recipients.extend(cc_list)
            if bcc_list:
                all_recipients.extend(bcc_list)
            
            self._send_via_smtp(msg, all_recipients)
            logger.info(f"邮件已成功发送至: {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"发送邮件失败: {str(e)}", exc_info=True)
            return False
    
    def _attach_file(
        self, 
        msg: MIMEMultipart, 
        file_path: str, 
        display_name: Optional[str] = None
    ):
        """添加附件"""
        try:
            display_name = display_name or os.path.basename(file_path)
            
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename= {display_name}')
            msg.attach(part)
            logger.info(f"已添加附件: {display_name}")
            
        except Exception as e:
            logger.error(f"添加附件失败: {str(e)}")
            raise
    
    def _send_via_smtp(self, msg: MIMEMultipart, recipients: List[str]):
        """通过SMTP发送邮件"""
        server = None
        try:
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            
            if self.config.use_tls:
                server.starttls()
            
            server.login(self.config.sender_email, self.config.sender_password)
            server.sendmail(self.config.sender_email, recipients, msg.as_string())
            
        finally:
            if server:
                server.quit()


def send_analysis_result(
    recipient_email: str,
    output_file_path: str,
    project_name: str = "GitHub License Analyzer",
    smtp_config: Optional[EmailConfig] = None
) -> bool:
    """
    发送分析结果邮件
    
    Args:
        recipient_email: 收件人邮箱
        output_file_path: 输出Excel文件路径
        project_name: 项目名称
        smtp_config: 邮件配置
        
    Returns:
        成功返回True，失败返回False
    """
    try:
        sender = EmailSender(smtp_config)
        
        subject = f"{project_name} - 分析结果"
        body = f"""
您好，

您的代码许可证分析已完成。请查看附件中的Excel文件获取详细结果。

分析时间: {formatdate(localtime=True)}
项目: {project_name}

此邮件由系统自动发送，请勿直接回复。

谢谢，
{project_name} 系统
"""
        
        return sender.send_email(
            recipient_email=recipient_email,
            subject=subject,
            body=body,
            attachment_path=output_file_path,
            attachment_name=os.path.basename(output_file_path)
        )
        
    except Exception as e:
        logger.error(f"发送分析结果邮件失败: {str(e)}", exc_info=True)
        return False
