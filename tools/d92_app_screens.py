"""Runs the real app window off-screen with training photos frozen in the
Tools tab and saves screenshots. Jobs/settings go to a temp dir."""
import os, sys, tempfile
from pathlib import Path
from unittest import mock
ROOT = Path('/home/aath/UniversalV1/UniversalV1'); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import cv2
from PySide6.QtWidgets import QApplication
APP = QApplication.instance() or QApplication([])
from app.theme import DARK_INDUSTRIAL_QSS
APP.setStyleSheet(DARK_INDUSTRIAL_QSS)  # the app's own black theme, as main() sets it
import app.settings as settings_mod, app.paths as paths_mod, app.jobs as jobs_mod
import app.tabs.settings_tab as settings_tab_mod
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "d92_app_screens"; out.mkdir(parents=True, exist_ok=True)
model = sys.argv[2] if len(sys.argv) > 2 else "weights/d92_quarter_glass_seg.pt"
tmp = Path(tempfile.mkdtemp())
patches = [mock.patch.object(settings_mod, 'APP_SETTINGS_FILE', tmp / 'app_settings.json'),
           mock.patch.object(paths_mod, 'MARK_MODEL_PATH', tmp / 'missing.pt'),
           mock.patch.object(jobs_mod, 'PART_CONFIGS_DIR', tmp / 'jobs'),
           mock.patch.object(jobs_mod, 'LAST_PART_FILE', tmp / 'jobs' / '.last_part'),
           mock.patch.object(settings_tab_mod, 'find_cameras', return_value=[]),
           mock.patch('app.tower.TowerController.connect_device'),
           mock.patch('app.main_window.MainWindow.confirm_unsaved', return_value=True),
           mock.patch('app.tabs.monitor_tab.MonitorTab.ensure_live', return_value=False)]
for p in patches: p.start()
from app.main_window import MainWindow
win = MainWindow(); win.ctx.set_current_part('D92_QUARTER_GLASS')
win.resize(1920, 1080); win.show(); win._show_tab("Tools")
for _ in range(5): APP.processEvents()
tab = win.tools_tab
tool = tab.add_tool('segment', (0.0, 0.0, 1.0, 1.0), name='D92 glass + sponge')
tool.model = model
tool.roi = (0.0, 0.0, 1.0, 1.0)  # the whole glass must be inside the tool's area
# (screenshot name or None, photo): a good glass shown back then front, then
# the next glass whose back has no sponges.
steps = [(None, '20261127_163221.jpg'), ('1_all_correct_PASS', '20261127_163334.jpg'),
         ('2_NG_no_sponge', '20261127_163116.jpg')]
for tag, name in steps:
    if tag and tag.startswith('2'):
        tool.reset_sides()  # a new glass (on the line: the glass left the view)
    tab.set_frame(cv2.imread(f"D92-Image/D92/{name}"))
    for _ in range(5): APP.processEvents()
    r = tab._results.get(tool.name)
    print(tag or '(back first)', name, '->', r.text if r else None, '|', tab.part_check_panel.banner.text(), tab.part_check_panel.rows())
    if tag:
        win.grab().save(str(out / f'{tag}.png'))
win.close()
