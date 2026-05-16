import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from smtp_config import *

def send_verification_email(to_email, username, verify_link):

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Confirm Your Email – Brainalyze"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email

    html = f"""
<!DOCTYPE html>
<html lang="en">
<body style="margin:0; padding:30px; background:#e9ecff; font-family:Poppins, Arial, sans-serif;">

  <table width="100%" align="center" style="max-width:520px; margin:auto; background:#ffffff; border-radius:16px; padding:35px;">

    <!-- Title -->
    <tr>
      <td style="text-align:center; font-size:26px; font-weight:700; color:#506DCA; padding-bottom:25px;">
        Brainalyze Email Confirmation
      </td>
    </tr>

    <!-- Greeting -->
    <tr>
      <td style="text-align:left; font-size:16px; color:#222; line-height:1.7;">
        Hi {username},
        <br><br>
        Please confirm your email by clicking the button below:
      </td>
    </tr>

    <!-- Button -->
    <tr>
      <td style="padding:30px 0; text-align:center;">
        <a href="{verify_link}" 
           style="background:#506DCA; color:white; padding:14px 32px; border-radius:12px;
                  text-decoration:none; font-size:16px; font-weight:600; display:inline-block;">
            Verify Email
        </a>
      </td>
    </tr>

    <!-- Footer -->
    <tr>
      <td style="text-align:left; font-size:13px; color:#777; padding-top:25px;">
        If you didn't request this email, you can safely ignore it.
      </td>
    </tr>

  </table>

</body>
</html>
"""


    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())

    return True
