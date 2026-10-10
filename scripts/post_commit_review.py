#!/usr/bin/env python3
"""Posts a commit review as comments on that commit on GitHub.

    scripts/post_commit_review.py <review.json> <commit> [--dry-run]

The review is codex-review.sh's output, in the shape its schema sets. It posts
one comment on the commit holding the whole review, then one comment on each
finding's line, where that line is part of the commit's diff: GitHub anchors a
commit comment to a position in the diff, not to a line of the file. A finding
on a line the commit did not touch stays in the summary alone, and the summary
says so. With --dry-run it prints what it would post and posts nothing.
"""
import json
import re
import subprocess
import sys


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def diff_positions(commit):
    """{path: {new-file line: diff position}} for every line the diff shows.

    A position counts lines down from a file's first hunk header: the line
    after it is 1, and each later hunk header takes a position of its own."""
    positions, path, position, line = {}, None, 0, 0
    # Prefixes, drivers and path quoting fixed here, not left to the user's git
    # configuration (diff.noprefix, external diff drivers, core.quotepath),
    # which would change the lines this parses.
    shown = run("git", "-c", "core.quotepath=false", "show", "--format=", "--unified=3", "--no-color",
                "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/", commit)
    for text in shown.splitlines():
        if text.startswith("diff --git "):
            path, position = None, 0
        elif text.startswith("+++ "):
            path = None if text == "+++ /dev/null" else text[len("+++ b/"):]
            positions.setdefault(path, {})
        elif path is not None and text.startswith("@@"):
            if position:
                position += 1   # a later hunk header has a position
            line = int(re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", text).group(1))
        elif path is not None and text[:1] in (" ", "+", "-", "\\"):
            position += 1   # "\ No newline at end of file" takes a position too
            if text[0] in (" ", "+"):
                positions[path][line] = position
                line += 1
    return positions


def summary(review, commit, unanchored):
    lines = [f"**Codex review of {commit[:7]}: {review['outcome']}**", "", review["summary"], ""]
    if review["findings"]:
        lines += ["### Findings", ""]
        for f in review["findings"]:
            where = f"`{f['path']}:{f['line']}`" if f["line"] else f"`{f['path']}`"
            cited = f" ({', '.join(f['guidelines'])})" if f["guidelines"] else ""
            kind = f" ({f['kind']})" if "kind" in f else ""
            lines += [f"- **{f['severity']}**{kind} {where} — {f['title']}{cited}"]
            if f in unanchored:
                lines += ["", "  " + f["body"].replace("\n", "\n  ")]
        lines += [""]
    # Reviews written before the prompt asked only for correctness and cost
    # carry the gate sections instead; whichever a review has is posted.
    for key, title in (("cost_walk", "Cost walk"), ("architecture_review", "C++ architecture review"),
                       ("performance_review", "C++ performance review"), ("mcp_grounding", "MCP grounding"),
                       ("residual_risk", "Residual risk")):
        if key in review:
            lines += [f"### {title}", "", review[key], ""]
    return "\n".join(lines)


def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]
    if len(args) != 2:
        sys.exit(__doc__)
    review = json.load(open(args[0]))
    commit = run("git", "rev-parse", args[1]).strip()
    repo = run("gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner").strip()
    positions = diff_positions(commit)

    anchored, unanchored = [], []
    for f in review["findings"]:
        position = positions.get(f["path"], {}).get(f["line"]) if f["line"] else None
        (anchored if position else unanchored).append((f, position) if position else f)

    posts = [{"body": summary(review, commit, unanchored)}]
    for f, position in anchored:
        cited = f"\n\nGuidelines: {', '.join(f['guidelines'])}" if f["guidelines"] else ""
        posts.append({"body": f"**{f['severity']}** — {f['title']}\n\n{f['body']}{cited}",
                      "path": f["path"], "position": position})

    for post in posts:
        if dry_run:
            print(json.dumps(post, indent=1))
            continue
        fields = [arg for k, v in post.items() for arg in ("-F" if k == "position" else "-f", f"{k}={v}")]
        run("gh", "api", f"repos/{repo}/commits/{commit}/comments", *fields)
    print(f"{'would post' if dry_run else 'posted'} {len(posts)} comments on {commit[:7]}: "
          f"the summary, and {len(anchored)} on lines of the diff", file=sys.stderr)


if __name__ == "__main__":
    main()
