# User Manual — Wire Harness Vision Inspection (UniversalV1)

**English** · [ภาษาไทย](USER_MANUAL_TH.md)

This manual is for line operators and supervisors who use the inspection station every day. The screen is a touch panel: tap the buttons with a finger. A mouse also works.

## Contents

1. [Daily start-up](#1-daily-start-up)
2. [Monitor tab — running the inspection](#2-monitor-tab--running-the-inspection)
3. [Understanding the result (OK / NG / STANDBY)](#3-understanding-the-result-ok--ng--standby)
4. [Job Change tab — switching part numbers](#4-job-change-tab--switching-part-numbers)
5. [Scanning a barcode](#5-scanning-a-barcode)
6. [Log tab — reviewing NG images](#6-log-tab--reviewing-ng-images)
7. [Settings tab (supervisor)](#7-settings-tab-supervisor)
8. [Capture and Train tab (supervisor)](#8-capture-and-train-tab-supervisor)
9. [Problems and what to do](#9-problems-and-what-to-do)

---

## 1. Daily start-up

1. Switch on the station. The app opens **full screen by itself** after about a minute.
2. The **last job used is loaded automatically**. Check the part number at the top left of the Monitor tab.
3. If the part number is wrong, change it in **Job Change** (section 4).
4. Press the blue **Start** button at the bottom right.

## 2. Monitor tab — running the inspection

![Monitor tab](images/monitor_running.png)

| Area | What it shows |
|---|---|
| Top strip, left | Current **part number** and **model** |
| Top strip, right | The result badge: **OK**, **NG** or **STANDBY** |
| Centre (amber frame) | Live camera picture with detection boxes |
| Right panel, **Detections** | Each detected class and how many were found |
| Right panel, **OK / NG** tiles | How many OK and NG cycles have been counted |
| Bottom strip, left | **Capture** and **Buzzer On/Off** |
| Bottom strip, right | Running status and the **Start / Stop** button |

**Buttons**

- **Start** (blue) starts the camera and the inspection. The button turns red and becomes **Stop**, and the status shows **● Running**.
  - The first start after power-on can take about 10 seconds while the system prepares.
- **Stop** (red) stops the inspection and switches off the tower light. The button shows *Stopping…* for a moment.
- **Buzzer On / Off** mutes or unmutes the buzzer. When it is orange, the buzzer is on. The tower light keeps working when the buzzer is off.
- **Capture** saves the current picture for training (see section 8).

**OK / NG counters**

![Counters](images/side_panel.png)

- One count is added each time the result **changes to** OK or NG.
  - A part that stays OK is counted once.
  - Remove the part (STANDBY) before the next one, so each part is counted separately.
- The counts are kept when the station is switched off.
- To start a new shift or lot from zero, tap **Reset Count** and confirm with **Yes**.

## 3. Understanding the result (OK / NG / STANDBY)

| Badge | Tower light | Buzzer | Meaning |
|---|---|---|---|
| ![OK](images/banner_green.png) | 🟢 Green | Short beep | The required parts were found. **Pass.** |
| ![NG](images/banner_red.png) | 🔴 Red | Alarm sound | Something was detected, but it does not match what the job requires. **Reject.** |
| ![STANDBY](images/banner_yellow.png) | 🟡 Yellow | Silent | Nothing detected. Waiting for a part. |

- The result is **held for at least 1 second** before it can change again. The light does not flicker, and each result is shown long enough to see.
- When you see **NG**, set the part aside and check it. The NG picture is saved automatically in the **Log** tab.

## 4. Job Change tab — switching part numbers

![Job Change tab](images/job_change.png)

Each **job** stores the settings for one part number: the model, the detection filters and the target classes.

- **Current Part** at the top shows the active job.
- **Saved Jobs** shows one button per job, with the part number and its model name.
  - The **orange** button is the job in use.
  - **Tap a button to switch.** The job is applied immediately.
- **Scan Barcode...** (blue, bottom right) selects a job by scanning the part's barcode (section 5).

**Removing a job (supervisor)**

![Remove Job dialog](images/remove_job.png)

1. Tap **Remove Job...** at the bottom left.
2. Choose the job in **Job**. The current job is selected first.
3. To confirm, **type the part number exactly** as shown, including upper and lower case. The red **Remove** button only becomes active when the text matches.
4. Tap **Remove**, or **Cancel** to keep the job.

- Only the saved job is deleted. NG pictures for that part stay in the **Log** tab.
- If you remove the job that is in use, the inspection keeps its current settings until you choose another job. That job will no longer load automatically at start-up.
- A removed job cannot be restored. Create it again with **Scan Barcode... → Save**.

## 5. Scanning a barcode

> **Stop** the inspection on the Monitor tab first. The camera can only be used by one screen at a time.

![Scan dialog](images/scan_dialog.png)

1. Tap **Scan Barcode...** on the Job Change tab.
2. Hold the barcode in front of the camera, **lined up with the red line**.
3. When it is read, the part number appears in the large **Detected** box.
   - **A job exists for this part:** the app asks *"Found saved config… Apply?"*. Tap **Yes**.
   - **No job exists yet:** the app asks whether to save the **current settings** as a new job for this part. Tap **Yes** only if the current settings are correct for this part.
4. If the barcode cannot be read, type the part number in the lower box:
   - **Load** opens the saved job for that part number.
   - **Save** stores the current settings as the job for that part number (supervisor).
5. Tap **Close** when finished, then press **Start** on the Monitor tab.

## 6. Log tab — reviewing NG images

![Log tab](images/log.png)

- Every NG result saves a picture of the moment it happened, with the detection boxes.
- Pictures are **grouped by part number**. The part with the most recent NG is at the top.
- Each group shows the **5 newest NG pictures**, newest on the left, with date and time.
- **Refresh** reloads the list.

**Viewing a picture**: tap a thumbnail. It opens at about 85% of the screen.

![NG picture](images/log_popup.png)

| Action | Touch | Mouse / keys |
|---|---|---|
| Zoom in / out | Pinch with **2 fingers** | Mouse wheel, or **+ / −** |
| Move the picture | Drag with 1 finger | Drag |
| Back to fitted size | Double-tap, or **Fit** | Double-click |
| Close | **✕** at top right, or tap outside the picture | **Esc** |

![Zoomed NG picture](images/log_popup_zoom.png)

## 7. Settings tab (supervisor)

![Settings tab](images/settings.png)

> Changes on this tab take effect **immediately**. The detection settings are stored in a job only when you **Save** the job (section 5, step 4). Buzzer settings and hold time are saved automatically.

**Model**

- **Load .pt Model** selects the YOLO model file.

**Camera**

- Choose the camera. Tap **Scan Cameras** if it is not listed.

**Tower Light & Buzzer**

| Setting | Meaning |
|---|---|
| Tower Light: Connected / Not Connected | Connection status of the tower light |
| **Enable Buzzer** | Main buzzer switch. Same as the **Buzzer** button on the Monitor tab. |
| **NG alarm** + **Test** | The alarm sound used for NG. **Test** plays it once. |
| **Beep on OK** + time + **Test** | Short confirmation beep on OK (default 50 ms) |
| **State hold time** | Minimum time a result is held before it can change (default 1000 ms, 0 = off) |

**Detection Filters**

| Setting | Meaning |
|---|---|
| **Confidence threshold** | Detections below this confidence are ignored. Higher means stricter and fewer false detections. |
| **Max Detections** | Keeps only the N most confident detections (0 = no limit) |
| **Target Classes** + **OR / AND** | The class(es) that mean **OK**. **OR**: either target is enough. **AND**: both must be present. |
| **Classes to show** | Only ticked classes are displayed and used for the judgement |

## 8. Capture and Train tab (supervisor)

Use this to improve the model when it makes mistakes.

**Step 1 — Collect pictures**

- On the Monitor tab, while running, tap **Capture**.
- The picture and its detection boxes are saved in the `captures/` folder, with the images in `images` and the box labels in `labels`.

**Step 2 — Correct the boxes**

![Train tab](images/train.png)

1. **Stop** the Monitor, then open the **Train** tab.
2. **Select Folder...** and choose `captures/images`. The top strip shows the **Active Model** and the **Dataset**.
3. Choose a picture in the **Images** list, or use **◀ Prev / Next ▶**.
4. Tap a box on the picture. **Box Correction** shows the selected box.
5. Choose the right class in **Change class to**, then tap **Confirm**. The label file is updated at once.

**Step 3 — Train**

1. Tap **Start Training** (blue, bottom right).
   - Training runs 50 rounds (epochs), and progress is shown in the bottom strip.
   - It can take a long time. Do not switch off the station.
2. When it finishes, the new model's path is shown. It is in `captures/runs/train…/weights/best.pt`.
3. Load it in **Settings → Load .pt Model**, test it on the Monitor tab, and **Save** the job (section 5) to keep it.

## 9. Problems and what to do

| Problem | What to do |
|---|---|
| Screen shows *Tower Light: Not Connected* | Check the tower light's USB cable. Restart the station if needed. |
| **Start** does nothing, or the picture stays black | Wait about 10 seconds on the first start. If it stays black, check the camera cable, then **Settings → Scan Cameras**. |
| *"Stop the Monitor stream before scanning"* | Press **Stop** on the Monitor tab, then open **Scan Barcode...** again. |
| Barcode is not read | Hold it steady on the red line, closer or further away, with good light. Or type the number and tap **Load**. |
| No sound on NG | Check that **Buzzer On** is orange on the Monitor tab, and that the tower light is connected. |
| Too many false NG | Tell the supervisor. Adjust the **Confidence threshold**, or retrain the model with **Capture** and **Train**. |
| The app closed by itself | It restarts automatically within a few seconds. If it keeps closing, tell maintenance. |
