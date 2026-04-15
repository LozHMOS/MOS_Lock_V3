# MOSLock V3 — Trial Site Demo Instructions
## BST008 Mobile Substation | Mapped Out Solutions® | mos | Lock®

**"We will always have a plan."**

---

## Before You Start

### Equipment Checklist
- [ ] Samsung Galaxy Tab S10 fully charged (>80%)
- [ ] Termux + MOSLock V3 installed and tested (see DEPLOYMENT_TAB_S10.md)
- [ ] Demo Mode tested offline — all 4 scenarios load without errors
- [ ] BST008 substation accessible and in the state you'll demo
- [ ] Site HV Permit in place (real permit for any live equipment)
- [ ] Safety helmet, high-vis, safety boots on all attendees

### Pre-Demo Check (10 minutes before)
1. Launch MOSLock: open Termux → `cd ~/moslock && streamlit run moslock_v3.py`
2. Open Chrome → `http://localhost:8501`
3. Verify Demo Mode loads Substation Trained scenario
4. Check all 6 tabs navigate cleanly
5. Test High Contrast Mode toggle (for underground)
6. If demoing Live Mode: verify ONNX model loads (green dot in sidebar)

---

## Demo Walk-Through Script

**Audience:** Electrical engineers + safety managers + site management  
**Duration:** 20–30 minutes  
**Setting:** Ideally beside a BST008 substation or in a meeting room with the tablet

---

### Opening (2 minutes)

*Hold up the tablet facing the audience*

> "What you're looking at is MOSLock V3 — a verification assistant for high-voltage
> isolation. Its job is to give the isolation verifier an independent, AI-powered
> check that every isolation point is in the right state before a worker enters
> a compartment."

> "MOSLock doesn't replace your 12-step process or your permit. It's the digital
> verification step at Step 5 — 'test for dead' — giving you a second set of eyes
> that never gets complacent."

> "Everything we built follows GCAA Fatal Hazard Protocol 7 and AS/NZS 4836."

---

### Tab 1: Detection (5 minutes)

*Select "Substation — Isolated LV" from the Demo Image selector in the sidebar*

> "Here's the BST008 from the V2 trial. The AI has already scanned it and
> identified several things."

Point to the bounding boxes:
- **Green boxes (isolation points):** "These are the isolation points in an OPEN/safe state — isolator open, circuit breaker open, disconnect open."
- **Cyan boxes (compartments):** "These are the three compartment zones — HV Incoming, Transformer, LV Outgoing."
- **Orange box (personal lock):** "This is a personal padlock detected at isolation point 3 — confirming someone has secured it."
- **Green indicator:** "This live line indicator is showing OFF — no voltage detected."

> "Each detection has a confidence score. We default to 90% — any detection
> below that threshold automatically triggers a REVIEW, not a pass."

*Pause for questions*

---

### Tab 2: Verification (5 minutes)

Click the **Verification** tab.

> "This is the engine room. The IFTTT rule set — what we call the Training Model —
> defines what a correctly isolated substation looks like. We compare every
> detection against those rules."

Point to the **verdict banner**:
- ✅ ALL CLEAR → "All isolation points verified, all rules passed."
- ❌ DISCREPANCY → "Something's wrong — don't enter."
- ⚠ REVIEW → "Needs a manual check."
- 🛑 STOP → "indicator_fault — mandatory manual test instrument check."

> "The rules are set by the site electrical engineer. You can see them in the
> Training Model tab — and you can add custom rules for your specific equipment."

Point to the **lock count KPI**:
> "MOSLock counts detected personal locks and compares to the number of workers
> signed on via the permit. If the numbers don't match — you get a flag."

> "This is the digital equivalent of the permit controller counting tags on
> the lock box before anyone enters."

*Ask: "What happens if an indicator is damaged?" → Point to indicator_fault STOP rule*

---

### Tab 3: Compartments (3 minutes)

Click the **Compartments** tab.

> "Ten compartment zones on a BST008. Green means isolated, red means energised,
> amber means we couldn't determine the state — needs a manual check."

> "In a full training model, each compartment is mapped to its isolation point.
> A compartment only goes green when its isolation point is detected as OPEN
> AND a lock is present AND the indicator is OFF."

*Change the demo scenario to "Substation — Base State" using the sidebar*

> "Compare this to the base state — no isolation detected, so all compartments
> show as unknown. No entry."

---

### Tab 4: Training Model (3 minutes)

Click the **Training Model** tab.

> "This is where an authorised electrical engineer configures what MOSLock
> expects to see. Each rule is: 'For this compartment, I need THIS isolation
> point in THIS state, with a lock, and the indicator showing OFF.'"

> "Rules are locked down for site use. Adding a rule requires the authorised
> engineer to open this tab and configure it. It's version-controlled — you
> know exactly what was in the training model at the time of each verification."

*Point to "Add Rule" expander*
> "Future versions will lock this behind an authorised engineer sign-in."

---

### Tab 5: Permit (3 minutes)

Click the **Permit** tab.

> "The permit register. Workers sign on using the form — their name, contact
> number, and the time. MOSLock records it. When they sign off, the timestamp
> is captured."

> "This mirrors the Part 11 sign-on/sign-off sheet in your GCAA HV permit.
> Eventually, MOSLock exports to your permit management system directly."

Point to 12-step progress:
> "And we track where you are in the 12-step process. You can see which steps
> are complete, which is active, and what's coming next."

*If a discrepancy exists, point to:*
> "Notice the system has flagged that 2 workers are signed on but only 1 lock
> is detected. That's an automatic hold — you can't sign off as ALL CLEAR until
> that's reconciled."

---

### Live Mode Demo (if model available — 5 minutes)

*Toggle Live Mode in sidebar*

> "If the ONNX model is loaded — this is the 6 megabyte file that contains
> 100 epochs of training — we switch to live inference."

*Point tablet camera at substation or equipment on a table*

> "Point. Capture. The model runs on the tablet's own processor — no internet,
> no cloud. We get detections in under a second."

> "At the trial site, we'll train the model on photos of THIS substation — your
> specific equipment, your locks, your indicator panel. That's when accuracy
> goes from 'good' to 'excellent'."

*Demonstrate confidence threshold slider*

> "This slider is the safety dial. At 90%, if the model isn't sure, it flags
> it for review. You can go higher — 95% — for maximum assurance. You'd never
> go below 80% for isolation verification."

---

### Closing (2 minutes)

> "The headline: MOSLock is a verification aid that speaks the language of
> your permit system, runs on a $700 tablet, works underground with no internet,
> and takes less than a second to scan an isolation point."

> "It doesn't replace your electrical engineer. It doesn't replace your permit
> issuer. It gives your isolation verifier a consistent, documented, AI-powered
> second opinion at the most critical moment — before someone enters a potentially
> live compartment."

> "We will always have a plan."

---

## FAQ for Electrical Engineers

**Q: What if the model gets a detection wrong — false positive on a circuit breaker state?**  
A: The confidence threshold prevents low-confidence detections from contributing to an ALL CLEAR verdict. Any detection below 90% (default) flags the verification as REVIEW REQUIRED, forcing a manual check. A wrong detection generates a DISCREPANCY or REVIEW, never a false ALL CLEAR. The system fails safe.

**Q: How long does training take? We have 50 site photos.**  
A: With 50 annotated real photos (using bootstrap_annotations.py for pre-annotation) + 750 synthetic images, training takes approximately 45 minutes on a laptop with a mid-range GPU, or 4–6 hours on CPU. After training, the exported ONNX file is 6 MB and loads in 1–2 seconds on the Tab S10.

**Q: What happens to detections from a different substation type?**  
A: The model is trained per substation type. BST008 training data produces a BST008 model. For a different make/model of substation, you create a new dataset and train a new model (or fine-tune from the BST008 model using transfer learning — typically faster). The Training Model (IFTTT rules) is also configured per substation type.

**Q: Does this comply with AS/NZS 4836 requirements for isolation verification?**  
A: MOSLock is a supplementary tool that supports — but does not replace — the required physical verification procedures in AS/NZS 4836:2023. The 12-step GCAA process must still be followed in full. MOSLock provides documented evidence of the visual verification state at a point in time, which supports audit trails.

**Q: Can the app be tampered with to bypass safety checks?**  
A: The verdict logic is in open-source Python (moslock_v3.py). For production deployment, the Training Model rules would be digitally signed and locked, with audit logs stored off-device. This is on the V3.1 roadmap.

**Q: What is the latency on the Tab S10?**  
A: Camera capture to verdict: approximately 1.5–3 seconds (capture 0.5s + inference 0.8s + rendering 0.5s). For a substation with 3 isolation points, a full verification takes under 10 seconds.

---

## FAQ for Safety Managers

**Q: If a worker doesn't sign off at end of shift, does that create an issue?**  
A: Yes — the lock count will show a discrepancy (locks detected may not match workers on register). The system flags this as REVIEW REQUIRED on the next verification scan. This is intentional — it forces reconciliation.

**Q: What happens if the tablet battery dies mid-verification?**  
A: The permit register data is stored in Streamlit session state (in-memory). A battery death loses the session. For production, session state would be persisted to a local SQLite database or synced to a cloud permit management system. This is on the roadmap.

**Q: Can we see a history of verifications performed?**  
A: Not in V3.0 — each scan is a point-in-time verification. V3.1 roadmap includes: timestamped scan history with annotated images, exportable PDF/CSV audit report, and integration with site permit management systems.

**Q: Is this appropriate for use in an underground mine?**  
A: The tablet (Samsung Galaxy Tab S10) is an intrinsically non-safe device and must be used in accordance with site electrical equipment rules for the specific area classification. The High Contrast Mode is specifically designed for low-light underground environments. Battery life in Live Mode is approximately 6 hours — a power bank is recommended for full-shift use.

**Q: Who is responsible for maintaining the Training Model (IFTTT rules)?**  
A: The site's Principal Electrical Engineer (or their delegate, as per the GCAA HV Permit structure). Only authorised engineers should modify the rule set. In production, rule changes are version-controlled and require digital sign-off.

**Q: Can MOSLock detect if someone has tampered with a lock?**  
A: Currently, MOSLock detects the presence and type of locks but does not assess their integrity (e.g., cut shackle, damaged lock). Physical inspection of lock integrity remains a human responsibility per the 12-step process.

**Q: What training is required for operators?**  
A: Operators need to understand: (1) Demo Mode vs Live Mode, (2) how to interpret the verdict banner, (3) that REVIEW and STOP verdicts require manual physical verification, (4) how to use the sign-on register. Estimated training time: 30 minutes hands-on with a trained MOSLock operator.

---

## Demo Scenarios Summary

| Scenario | What it Shows | Expected Verdict |
|----------|--------------|-----------------|
| Substation — Base State | No isolation — baseline before work starts | REVIEW (no isolation detected) |
| Substation — Training Model | AI detections overlaid, 3 isolation points visible | REVIEW (rules partially met) |
| Substation — Isolated LV | LV compartment isolated, lock present, indicator off | REVIEW → ALL CLEAR (with workers signed on) |
| Substation — Locked Out | Lock detected at isolation point 3 | REVIEW (partial isolation) |

For a full ALL CLEAR demo: load "Substation — Isolated LV", set Workers Signed On = 1 in sidebar, set confidence threshold to 85%, and confirm all rules show PASS in the Verification tab.

---

*Mapped Out Solutions® | mos | Lock® V3.0 | "We will always have a plan."*  
*Demo prepared: December 2025 | Contact: mappedoutsolutions.com*
