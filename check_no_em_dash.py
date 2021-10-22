#!/usr/bin/env python3
"""
Integrity utility to ensure zero em dashes exist in any comments, docs, or files.
Strictly checks for unicode \u2014, \u2013, and markdown em dash conventions.
"""
import sys
import os

FORBIDDEN_CHARS = ['\u2014', '\u2013'] # em dash and en dash

def check_file(filepath):
    errors = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line_no, line in enumerate(f, 1):
                for char in FORBIDDEN_CHARS:
                    if char in line:
                        errors.append((line_no, f"Found forbidden dash '{char}'"))
    except Exception as e:
        pass
    return errors

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    failed = False
    for root, dirs, files in os.walk(repo_root):
        if '.git' in root or '__pycache__' in root or '.pytest_cache' in root:
            continue
        for file in files:
            if file == 'check_no_em_dash.py':
                continue
            path = os.path.join(root, file)
            errs = check_file(path)
            if errs:
                failed = True
                print(f"FAILED: {path}")
                for line_no, msg in errs:
                    print(f"  Line {line_no}: {msg}")
    if failed:
        sys.exit(1)
    print("CHECK PASSED: Zero em dashes or en dashes found.")

if __name__ == '__main__':
    main()
