#!/usr/bin/env python3
"""Send deployment notification emails using project's SMTP config."""
import os
import sys
import smtplib
import subprocess
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate

PROJECT_DIR = Path(__file__).parent.parent

# Load .env manually to avoid external dep at this stage
def load_env():
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

def git_info():
    try:
        run = lambda *args: subprocess.check_output(list(args), cwd=PROJECT_DIR, stderr=subprocess.DEVNULL).decode().strip()
        return {
            "hash":   run("git", "rev-parse", "--short", "HEAD"),
            "msg":    run("git", "log", "-1", "--pretty=%s"),
            "author": run("git", "log", "-1", "--pretty=%an <%ae>"),
            "time":   run("git", "log", "-1", "--pretty=%ci"),
        }
    except Exception:
        return {"hash": "unknown", "msg": "unknown", "author": "unknown", "time": "unknown"}

def send(trigger: str, status: str, detail: str = ""):
    load_env()

    smtp_server   = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port     = int(os.getenv("SMTP_PORT", "587"))
    sender_email  = os.getenv("SENDER_EMAIL")
    sender_pass   = os.getenv("SENDER_PASSWORD")
    notify_email  = os.getenv("NOTIFY_EMAIL") or sender_email

    if not sender_email or not sender_pass:
        print("Email credentials not configured, skipping notification.", file=sys.stderr)
        return

    info = git_info()
    icon = "✅" if status == "success" else "❌"
    status_cn = "成功" if status == "success" else "失败"
    subject = f"{icon} LicenseDetector 部署{status_cn} [{info['hash']}] — {trigger}"

    body = f"""\
LicenseDetector 自动部署通知
{"=" * 50}

触发方式 : {trigger}
部署状态 : {status.upper()}
通知时间 : {formatdate(localtime=True)}

Git 信息
  提交哈希 : {info['hash']}
  提交消息 : {info['msg']}
  提交作者 : {info['author']}
  提交时间 : {info['time']}
{"" if not detail else f"""
详情
{detail}
"""}
服务地址 : http://0.0.0.0:8000
"""

    msg = MIMEMultipart()
    msg["From"]    = sender_email
    msg["To"]      = notify_email
    msg["Date"]    = formatdate(localtime=True)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        srv = smtplib.SMTP(smtp_server, smtp_port, timeout=15)
        srv.starttls()
        srv.login(sender_email, sender_pass)
        srv.sendmail(sender_email, [notify_email], msg.as_string())
        srv.quit()
        print(f"Notification sent → {notify_email}")
    except Exception as exc:
        print(f"Failed to send notification: {exc}", file=sys.stderr)

if __name__ == "__main__":
    trigger = sys.argv[1] if len(sys.argv) > 1 else "manual"
    status  = sys.argv[2] if len(sys.argv) > 2 else "success"
    detail  = sys.argv[3] if len(sys.argv) > 3 else ""
    send(trigger, status, detail)
