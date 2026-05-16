import os

os.environ["REPORT_LLM_URL"] = "https://joudnaif-llm.hf.space"

from shared import app

import routes.auth
import routes.home
import routes.patients
import routes.scans
import routes.reports
import routes.profile

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)