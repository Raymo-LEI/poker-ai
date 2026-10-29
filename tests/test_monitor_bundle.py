"""The allowlisted bundle boots after extraction with site packages disabled."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import ProxyHandler, build_opener
import zipfile


def test_standalone_bundle_runs_without_training_checkout_or_site_packages(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / 'monitor.zip'
    subprocess.run([sys.executable, str(repo/'scripts/package_monitor.py'), '--output', str(output)], check=True)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read('overwatch-monitor/MANIFEST.json'))
        for entry in manifest['files']:
            data = archive.read('overwatch-monitor/' + entry['path'])
            assert len(data) == entry['bytes']
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
        assert not any('/src/' in name or name.endswith(('.pt','.jsonl')) for name in archive.namelist())
        archive.extractall(tmp_path/'extracted')
    root = tmp_path/'extracted/overwatch-monitor'
    # Fresh copy has neither alphanl nor training dependency files.
    assert not (root/'src').exists()
    run = tmp_path/'logs/candidate'
    run.mkdir(parents=True)
    (run/'metrics.csv').write_text('iteration,p0_loss\n1,3.5\n')
    process = subprocess.Popen([sys.executable,'-I','-S','scripts/monitor_training.py',
        '--runs-dir',str(run.parent),'--port','0'],cwd=root,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        line = process.stdout.readline()
        assert re.match(r'Overwatch: http://127.0.0.1:\d+',line), line
        url = line.strip().split(' ',1)[1]
        opener = build_opener(ProxyHandler({}))
        with opener.open(url+'/api/run?id=0%2Fcandidate',timeout=5) as response:
            assert json.load(response)['latest']['train/loss_p0']==3.5
        for name in ('index.html','app.js','workbench.js','styles.css','workbench.css'):
            with opener.open(url+'/'+name,timeout=5) as response:
                assert response.status==200 and response.read()
        subprocess.run([sys.executable,'-I','-S','scripts/monitor_remote.py','--help'],
                       cwd=root,check=True,capture_output=True,text=True)
    finally:
        process.terminate()
        process.communicate(timeout=5)
