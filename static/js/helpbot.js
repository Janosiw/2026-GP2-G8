(function () {
  const QA = [
    {
      patterns: ["change password", "reset password", "forgot password"],
      answer: "To change your password: go to <b>Profile</b> from the top menu, then click the 🔒 lock icon and follow the steps.",
      chips: ["How do I edit my profile?", "Where is the Dashboard?"]
    },
    {
      patterns: ["edit profile", "update profile", "my profile", "profile"],
      answer: "To edit your profile: go to <b>Profile</b> from the top menu and click the ✏️ pencil icon to update your information.",
      chips: ["How do I change my password?", "Where is the Dashboard?"]
    },
    {
      patterns: ["add patient", "new patient", "create patient", "how do i add a patient", "how to add patient"],
      answer: "To add a new patient: go to the <b>Patients</b> page from the sidebar, then click <b>+ Add Patient</b> and fill in the patient details.",
      chips: ["How do I add a case?", "How do I upload an MRI?"]
    },
    {
      patterns: ["share a case", "share case", "invite colleague", "collaborate", "invite"],
      answer: "Currently, case sharing is not available. Each doctor manages their own patients and cases independently within Brainalyze.",
      chips: ["How do I generate a report?", "Where is the Dashboard?"]
    },
    {
      patterns: ["add case", "new case", "create case", "how do i add a case", "how to create case"],
      answer: "To create a new case: open the patient's profile, then click <b>+ New Case</b>. Set the start date and you will be taken to the scan analysis page.",
      chips: ["How do I upload an MRI?", "How do I generate a report?"]
    },
    {
      patterns: ["upload mri", "upload scan", "how do i upload", "mri scan", "nifti", ".nii", ".zip", "drag and drop"],
      answer: "To upload an MRI scan: open a case and you'll be taken to the analysis page. Click the upload area or drag and drop your file.<br>Supported formats: <b>PNG, JPG</b> (2D) · <b>NIfTI .nii/.nii.gz</b> (3D) · <b>ZIP</b> containing a NIfTI file.",
      chips: ["What is the difference between 2D and 3D?", "How do I generate a report?"]
    },
    {
      patterns: ["2d vs 3d", "2d and 3d", "difference between", "what is 2d", "what is 3d", "3d scan", "2d scan"],
      answer: "<b>2D (image):</b> PNG or JPG — the tumor type is classified and 2D segmentation is applied.<br><b>3D (NIfTI/ZIP):</b> .nii or ZIP file — the full tumor volume is segmented using nnUNet and volume is calculated in cm³.",
      chips: ["How do I upload an MRI?", "How do I view Tumor Progress?"]
    },
    {
      patterns: ["generate report", "download report", "create report", "view report", "report"],
      answer: "To generate a report: open the scan result page, then click <b>Generate Report</b>. You can view all reports from the <b>Reports</b> section in the sidebar.",
      chips: ["Where is the Dashboard?", "How do I view Tumor Progress?"]
    },
    {
      patterns: ["dashboard", "analytics", "statistics", "charts", "overview"],
      answer: "The <b>Dashboard</b> is accessible from the sidebar. It shows patient statistics, recovery rates, tumor type breakdown, and monthly trends.",
      chips: ["How do I view Tumor Progress?", "How do I generate a report?"]
    },
    {
      patterns: ["tumor progress", "scan progress", "baseline", "growth", "shrinkage", "scan comparison"],
      answer: "Tumor Progress is calculated automatically when a case has more than one scan:<br>• First scan = <b>Baseline</b><br>• Later scans show % change compared to the baseline (+growth / −shrinkage).",
      chips: ["What is the difference between 2D and 3D?", "How do I generate a report?"]
    },
    {
      patterns: ["error", "not working", "issue", "failed", "problem", "bug"],
      answer: "If you're experiencing a technical issue:<br>• Make sure the uploaded file format is supported<br>• Refresh the page and try again<br>• Contact the administrator if the problem persists.",
      chips: ["How do I upload an MRI?", "Where is the Dashboard?"]
    }
  ];

  const SUGGESTIONS = [
    "How do I add a patient?",
    "How do I add a case?",
    "How do I upload an MRI?",
    "2D vs 3D scans?",
    "How do I generate a report?",
    "Where is the Dashboard?"
  ];

  function findAnswer(text) {
    const t = text.toLowerCase();
    for (const qa of QA) {
      if (qa.patterns.some(p => t.includes(p.toLowerCase()))) {
        return qa;
      }
    }
    return null;
  }

  function injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
      #cb-btn {
        position: fixed; bottom: 28px; right: 28px; z-index: 9999;
        width: 62px; height: 62px; border-radius: 50%;
        background: linear-gradient(135deg, #3B82F6 0%, #506DCA 100%);
        box-shadow: 0 4px 20px rgba(59,130,246,0.45);
        cursor: pointer; border: none; outline: none;
        display: flex; align-items: center; justify-content: center;
        transition: transform .2s, box-shadow .2s;
      }
      #cb-btn:hover { transform: scale(1.08); box-shadow: 0 6px 28px rgba(59,130,246,0.6); }
      #cb-btn svg { width: 30px; height: 30px; fill: #fff; }
      #cb-tooltip {
        position: fixed; bottom: 100px; right: 28px; z-index: 9998;
        background: #fff; color: #374151; font-size: 13px; font-weight: 600;
        padding: 7px 14px; border-radius: 20px;
        box-shadow: 0 3px 14px rgba(0,0,0,0.13);
        pointer-events: none; opacity: 0; transition: opacity .25s;
        white-space: nowrap; font-family: 'Segoe UI', sans-serif;
      }
      #cb-btn:hover + #cb-tooltip { opacity: 1; }
      #cb-panel {
        position: fixed; bottom: 104px; right: 28px; z-index: 9997;
        width: 340px; max-height: 520px;
        background: #fff; border-radius: 18px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.18);
        display: flex; flex-direction: column;
        overflow: hidden; font-family: 'Segoe UI', sans-serif;
        transform: scale(.92) translateY(12px); opacity: 0;
        pointer-events: none; transition: transform .22s cubic-bezier(.34,1.36,.64,1), opacity .18s;
      }
      #cb-panel.open {
        transform: scale(1) translateY(0); opacity: 1; pointer-events: all;
      }
      #cb-header {
        background: linear-gradient(135deg, #3B82F6 0%, #506DCA 100%);
        padding: 14px 16px; display: flex; align-items: center; gap: 11px; color: #fff;
      }
      #cb-header svg { width: 36px; height: 36px; flex-shrink: 0; }
      #cb-header-text h4 { margin: 0; font-size: 14px; font-weight: 700; }
      #cb-header-text p { margin: 2px 0 0; font-size: 11px; opacity: .85; }
      #cb-close {
        margin-left: auto; background: rgba(255,255,255,.2); border: none;
        color: #fff; width: 26px; height: 26px; border-radius: 50%;
        cursor: pointer; font-size: 16px; display: flex; align-items: center;
        justify-content: center; flex-shrink: 0; line-height: 1;
      }
      #cb-close:hover { background: rgba(255,255,255,.35); }
      #cb-messages {
        flex: 1; overflow-y: auto; padding: 14px; display: flex;
        flex-direction: column; gap: 10px;
        scrollbar-width: thin; scrollbar-color: #e0e7ff transparent;
      }
      .cb-msg { display: flex; gap: 8px; align-items: flex-start; }
      .cb-msg.user { flex-direction: row-reverse; }
      .cb-bubble {
        max-width: 78%; padding: 9px 13px; border-radius: 14px;
        font-size: 13px; line-height: 1.55; word-break: break-word;
      }
      .cb-msg.bot .cb-bubble {
        background: #f3f4f6; color: #1f2937; border-top-left-radius: 4px;
      }
      .cb-msg.user .cb-bubble {
        background: linear-gradient(135deg, #3B82F6, #506DCA);
        color: #fff; border-top-right-radius: 4px;
      }
      .cb-avatar {
        width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
        background: linear-gradient(135deg,#3B82F6,#506DCA);
        display: flex; align-items: center; justify-content: center;
        font-size: 14px;
      }
      .cb-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
      .cb-chip {
        background: #dbeafe; color: #1d4ed8; border: none; border-radius: 999px;
        padding: 5px 12px; font-size: 11.5px; font-weight: 600; cursor: pointer;
        font-family: 'Segoe UI', sans-serif; transition: background .15s;
      }
      .cb-chip:hover { background: #bfdbfe; }
      #cb-suggestions {
        padding: 0 12px 10px; display: flex; flex-wrap: wrap; gap: 6px;
      }
      #cb-input-row {
        border-top: 1px solid #f3f4f6; padding: 10px 12px;
        display: flex; align-items: center; gap: 8px;
      }
      #cb-input {
        flex: 1; border: 1.5px solid #e5e7eb; border-radius: 999px;
        padding: 8px 14px; font-size: 13px; outline: none;
        font-family: 'Segoe UI', sans-serif; direction: ltr;
        transition: border-color .15s;
      }
      #cb-input:focus { border-color: #506DCA; }
      #cb-send {
        width: 36px; height: 36px; background: linear-gradient(135deg,#3B82F6,#506DCA);
        border: none; border-radius: 50%; color: #fff; cursor: pointer;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
        transition: opacity .15s;
      }
      #cb-send:hover { opacity: .88; }
      #cb-send svg { width: 16px; height: 16px; }
      #cb-pulse {
        position: fixed; bottom: 28px; right: 28px; z-index: 9996;
        width: 62px; height: 62px; border-radius: 50%;
        background: rgba(59,130,246,.3); pointer-events: none;
        animation: cb-pulse-anim 2s ease-out infinite;
      }
      @keyframes cb-pulse-anim {
        0% { transform: scale(1); opacity: .7; }
        70% { transform: scale(1.55); opacity: 0; }
        100% { transform: scale(1.55); opacity: 0; }
      }
      @media (max-width: 480px) {
        #cb-panel { width: calc(100vw - 24px); right: 12px; bottom: 96px; }
        #cb-btn, #cb-pulse { right: 16px; bottom: 20px; }
        #cb-tooltip { right: 16px; }
      }
    `;
    document.head.appendChild(style);
  }

  function buildWidget() {
    const botIcon = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C6.48 2 2 6.03 2 11c0 2.67 1.19 5.06 3.07 6.74L4 20l3.3-1.32A10.17 10.17 0 0012 20c5.52 0 10-4.03 10-9S17.52 2 12 2zm1 13H8v-2h5v2zm3-4H8V9h8v2z"/></svg>`;
    const sendIcon = `<svg viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>`;

    document.body.insertAdjacentHTML('beforeend', `
      <div id="cb-pulse"></div>
      <button id="cb-btn" aria-label="Help Assistant">${botIcon}</button>
      <div id="cb-tooltip">Need help?</div>
      <div id="cb-panel" role="dialog" aria-label="Help Assistant">
        <div id="cb-header">
          ${botIcon}
          <div id="cb-header-text">
            <h4>Brainalyze Assistant</h4>
            <p>How can I help you today?</p>
          </div>
          <button id="cb-close" aria-label="Close">✕</button>
        </div>
        <div id="cb-messages"></div>
        <div id="cb-suggestions"></div>
        <div id="cb-input-row">
          <input id="cb-input" type="text" placeholder="Type your question here..." autocomplete="off" />
          <button id="cb-send" aria-label="Send">${sendIcon}</button>
        </div>
      </div>
    `);

    const btn = document.getElementById('cb-btn');
    const panel = document.getElementById('cb-panel');
    const closeBtn = document.getElementById('cb-close');
    const messages = document.getElementById('cb-messages');
    const input = document.getElementById('cb-input');
    const sendBtn = document.getElementById('cb-send');
    const suggestionsEl = document.getElementById('cb-suggestions');
    const pulse = document.getElementById('cb-pulse');
    let opened = false;

    function togglePanel() {
      opened = !opened;
      panel.classList.toggle('open', opened);
      pulse.style.display = opened ? 'none' : '';
      if (opened && messages.children.length === 0) {
        showBotMessage('Hi! I\'m the Brainalyze Assistant 🧠<br>How can I help you today?', SUGGESTIONS);
      }
      if (opened) setTimeout(() => input.focus(), 250);
    }

    btn.addEventListener('click', togglePanel);
    closeBtn.addEventListener('click', togglePanel);

    function showBotMessage(html, chips) {
      const div = document.createElement('div');
      div.className = 'cb-msg bot';
      div.innerHTML = `
        <div class="cb-avatar">🤖</div>
        <div>
          <div class="cb-bubble">${html}</div>
          ${chips && chips.length ? `<div class="cb-chips">${chips.map(c => `<button class="cb-chip">${c}</button>`).join('')}</div>` : ''}
        </div>
      `;
      messages.appendChild(div);
      div.querySelectorAll('.cb-chip').forEach(c => {
        c.addEventListener('click', () => handleQuery(c.textContent.trim()));
      });
      messages.scrollTop = messages.scrollHeight;
    }

    function showUserMessage(text) {
      const div = document.createElement('div');
      div.className = 'cb-msg user';
      div.innerHTML = `<div class="cb-bubble">${text}</div>`;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
    }

    function showSuggestions(chips) {
      suggestionsEl.innerHTML = chips.map(c => `<button class="cb-chip">${c}</button>`).join('');
      suggestionsEl.querySelectorAll('.cb-chip').forEach(c => {
        c.addEventListener('click', () => handleQuery(c.textContent.trim()));
      });
    }

    function handleQuery(text) {
      suggestionsEl.innerHTML = '';
      showUserMessage(text);
      setTimeout(() => {
        const qa = findAnswer(text);
        if (qa) {
          showBotMessage(qa.answer, qa.chips || []);
        } else {
          showBotMessage(
            "Sorry, I didn't quite understand that 😅<br>Try asking about: patients, cases, uploading MRI, reports, sharing, or the dashboard.",
            ['How do I upload an MRI?', 'How do I add a patient?', 'Where is the Dashboard?']
          );
        }
      }, 350);
    }

    sendBtn.addEventListener('click', () => {
      const txt = input.value.trim();
      if (!txt) return;
      input.value = '';
      handleQuery(txt);
    });

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') sendBtn.click();
    });

    showSuggestions(SUGGESTIONS.slice(0, 4));
  }

  function init() {
    injectStyles();
    buildWidget();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
