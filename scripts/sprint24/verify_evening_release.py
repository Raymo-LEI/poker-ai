"""Print the actual Vn and release from Botzone replay debug evidence."""
import argparse
import json
import re
import urllib.request
import urllib.parse
from pathlib import Path


def extract(raw):
    if isinstance(raw,dict):
        for k,v in raw.items():
            if k=='debug' and isinstance(v,str):yield v
            else:yield from extract(v)
    elif isinstance(raw,list):
        for x in raw:yield from extract(x)
    elif isinstance(raw,str):
        try:yield from extract(json.loads(raw))
        except (ValueError,TypeError):
            if raw.startswith('py'):yield raw


def parse_replay(text):
    try:return json.loads(text)
    except ValueError:
        assignment=re.search(r"\b_rawLogJSON\s*=\s*",text)
        if not assignment:raise ValueError('Replay HTML does not expose _rawLogJSON')
        raw,_=json.JSONDecoder().raw_decode(text[assignment.end():])
        if isinstance(raw,str):
            if '[Truncated]' in raw:raise ValueError('Replay log is truncated')
            raw=json.loads(raw)
        return raw


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--url');g.add_argument('--file',type=Path);p.add_argument('--expected')
    a=p.parse_args()
    if a.file:raw=parse_replay(a.file.read_text())
    else:
        parsed=urllib.parse.urlsplit(a.url)
        if (parsed.scheme!='https' or parsed.hostname not in ('botzone.org.cn','www.botzone.org.cn') or
            not re.fullmatch(r'/match/[a-fA-F0-9]{24}/?',parsed.path)):
            raise SystemExit('Expected an HTTPS Botzone match URL')
        req=urllib.request.Request(a.url,headers={'User-Agent':'Mozilla/5.0 (Evening release verification)'})
        with urllib.request.urlopen(req,timeout=30) as response:
            raw=parse_replay(response.read(64*1024*1024).decode('utf-8'))
    found=set()
    for debug in extract(raw):
        m=re.search(r'(V\d+)_MODEL\.npz.*?release=([^\s]+)',debug)
        if m:found.add((m.group(1),m.group(2)))
    if not found:raise SystemExit('No versioned model-load debug found; runtime release is unverified')
    for version,release in sorted(found):print(version,'release='+release)
    if a.expected and any(r!=a.expected for v,r in found):raise SystemExit('Runtime release mismatch')

if __name__=='__main__':main()
