import os

os.environ["REPORT_LLM_URL"] = "https://joudnaif-llm.hf.space"

import shared

import routes.auth
import routes.home
import routes.patients
import routes.scans
import routes.reports
import routes.profile

if __name__ == "__main__":
    shared.app.run(debug=True, host='0.0.0.0', port=5000)
