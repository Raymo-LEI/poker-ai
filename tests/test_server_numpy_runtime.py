"""NumPy inference must import a real stage2_runtime zip in any member order.

``server_service.Service`` starts ``server_numpy_runtime.NumpyBundle`` before it
accepts a request. The checked-in Botzone export lists ``allin_mask.py`` before
``equity.py``. Executing that order used to raise ``ModuleNotFoundError`` for
the relative import and every later benchmark gate died with it.
"""
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_SOURCE = ROOT / 'botzone_bots' / 'stage2.zip'
SPRINT = ROOT / 'scripts' / 'sprint24'
CHILD = r'''
import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import server_numpy_runtime
from server_numpy_runtime import NumpyBundle

assert Path(server_numpy_runtime.__file__).name == 'server_numpy_runtime.py'
mode, missing = sys.argv[2], sys.argv[3]
zips = sys.argv[4:]

def package_name(path):
    return '_server_numpy_' + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

if mode == 'load':
    for zip_path in zips:
        try:
            NumpyBundle(zip_path, missing)
        except FileNotFoundError:
            pass
        else:
            raise SystemExit('model load was not reached for ' + zip_path)
        name = package_name(zip_path)
        mask = sys.modules[name + '.allin_mask']
        equity = sys.modules[name + '.equity']
        guard = sys.modules[name + '.deployment_guard']
        runtime = sys.modules[name + '.runtime']
        cfv = sys.modules[name + '.cfv']
        solver = sys.modules[name + '.deep_solver_v2']
        assert mask.estimated_equity is equity.estimated_equity
        assert mask.preflop_key is equity.preflop_key
        assert mask.preflop_key(('SA', 'HK')) == 'AKo'
        assert guard.strength is sys.modules[name + '.portable_evaluator'].strength
        assert guard.blind_budget is sys.modules[name + '.blind_lock'].blind_budget
        assert runtime.mask_snapshot is mask.mask_snapshot
        assert runtime.estimated_equity is equity.estimated_equity
        with zipfile.ZipFile(zip_path) as archive:
            assert cfv.__loader__.get_data(cfv.__file__) == archive.read('stage2_runtime/cfv.py')
            assert cfv.__dict__['__loader__'].get_data(cfv.__file__) == archive.read('stage2_runtime/cfv.py')
            assert solver.__loader__.get_data(solver.__file__) == archive.read('stage2_runtime/deep_solver_v2.py')
        again = sys.modules[name + '.allin_mask']
        try:
            NumpyBundle(zip_path, missing)
        except FileNotFoundError:
            pass
        assert sys.modules[name + '.allin_mask'] is again
elif mode == 'reject':
    zip_path = zips[0]
    name = package_name(zip_path)
    try:
        NumpyBundle(zip_path, missing)
    except ModuleNotFoundError as exc:
        assert name + '.equity' in str(exc)
    else:
        raise SystemExit('zip without equity was accepted')
    assert name not in sys.modules
    assert not any(key.startswith(name + '.') for key in sys.modules)
else:
    raise SystemExit('unknown mode ' + mode)
if 'torch' in sys.modules:
    raise SystemExit('NumPy service imported torch')
print('ok')
'''


def _repack(source, destination, names):
    with zipfile.ZipFile(source) as archive:
        payload = {name: archive.read(name) for name in names}
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            out.writestr(info, payload[name])


def _run(tmp_path, mode, zips):
    script = tmp_path / 'load_numpy_bundle.py'
    script.write_text(CHILD)
    missing = tmp_path / 'missing-model.npz'
    result = subprocess.run(
        [sys.executable, str(script), str(SPRINT), mode, str(missing), *[str(path) for path in zips]],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert result.stdout.strip() == 'ok'


def test_numpy_bundle_loads_real_export_when_equity_follows_allin_mask(tmp_path):
    """Alphabetical and reversed copies of the shipped source zip both import."""
    with zipfile.ZipFile(REAL_SOURCE) as archive:
        names = archive.namelist()
        mask = archive.read('stage2_runtime/allin_mask.py').decode()
    runtime = [name for name in names if name.startswith('stage2_runtime/') and name.endswith('.py')]
    assert runtime == sorted(runtime)
    assert runtime.index('stage2_runtime/allin_mask.py') < runtime.index('stage2_runtime/equity.py')
    assert 'from .equity import estimated_equity, preflop_key' in mask
    reversed_zip = tmp_path / 'reversed.zip'
    _repack(REAL_SOURCE, reversed_zip, list(reversed(names)))
    with zipfile.ZipFile(REAL_SOURCE) as original, zipfile.ZipFile(reversed_zip) as archive:
        reversed_runtime = [name for name in archive.namelist() if name.startswith('stage2_runtime/') and name.endswith('.py')]
        assert archive.read('stage2_runtime/allin_mask.py') == original.read('stage2_runtime/allin_mask.py')
        assert archive.read('stage2_runtime/equity.py') == original.read('stage2_runtime/equity.py')
    assert reversed_runtime.index('stage2_runtime/equity.py') < reversed_runtime.index('stage2_runtime/allin_mask.py')
    _run(tmp_path, 'load', [REAL_SOURCE, reversed_zip])


def test_peak_rss_ignores_parent_high_water_after_exec():
    if not Path('/proc/self/status').exists():
        import pytest
        pytest.skip('Linux VmHWM is the production RSS source')
    blob=bytearray(256*1024*1024)
    env=dict(os.environ)
    env['PYTHONPATH']=str(SPRINT)+os.pathsep+env.get('PYTHONPATH','')
    result=subprocess.run([sys.executable,'-c','from server_numpy_runtime import peak_rss_mb; print(peak_rss_mb())'],capture_output=True,text=True,env=env)
    del blob
    assert result.returncode==0, result.stderr
    assert float(result.stdout.strip())<180


def test_numpy_bundle_drops_a_partial_package_when_equity_is_absent(tmp_path):
    """A missing relative-import target fails closed and leaves no ready package."""
    with zipfile.ZipFile(REAL_SOURCE) as archive:
        names = [name for name in archive.namelist() if name != 'stage2_runtime/equity.py']
    broken = tmp_path / 'missing-equity.zip'
    _repack(REAL_SOURCE, broken, names)
    _run(tmp_path, 'reject', [broken])
