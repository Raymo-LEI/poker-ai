"""Package the complete standalone Overwatch monitor using only stdlib."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'scripts/monitor_training.py', 'scripts/monitor_workbench.py', 'scripts/monitor_remote.py',
    'scripts/package_monitor.py', 'scripts/experiment_metadata.py', 'scripts/label_experiment.py', 'tools/training_monitor/index.html',
    'tools/training_monitor/app.js', 'tools/training_monitor/workbench.js',
    'tools/training_monitor/styles.css', 'tools/training_monitor/workbench.css',
    '打开训练监控.command', '连接服务器监控.command',
    'monitor/README.md', 'docs/training_monitor.md',
)


def package(output):
    entries = []
    for name in FILES:
        path = ROOT / name
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'missing or external monitor source: {name}')
        body = path.read_bytes()
        entries.append({'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(ROOT / entry['path'], 'overwatch-monitor/' + entry['path'])
        archive.writestr('overwatch-monitor/README.md', (ROOT / 'monitor/README.md').read_bytes())
        archive.writestr('overwatch-monitor/MANIFEST.json', json.dumps(
            {'format': 'overwatch_monitor_bundle_v1', 'files': entries}, indent=2) + '\n')
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/overwatch-monitor.zip')
    args = parser.parse_args()
    print(package(args.output).resolve())


if __name__ == '__main__':
    main()
