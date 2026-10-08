# Inherited Governance

This repo inherits its engineering philosophies, C++ architecture and
performance gates, and workflow from `plainsight-systems-governance`, checked
out in-tree as a submodule at:

**[`docs/process/governance/`](./governance/)**

Canonical source: <https://github.com/plainsight-systems/plainsight-systems-governance>
(public, CC BY 4.0). Its README is the index.

## Why a submodule

The documents are pinned to a specific commit, so it is always recoverable
which version of the review gates a given change was written and reviewed
against. A floating link cannot answer that.

Clone with the documents present:

```sh
git clone --recurse-submodules <this repo>
```

Already cloned:

```sh
git submodule update --init
```

## Updating

The pin advances deliberately, not automatically:

```sh
git submodule update --remote docs/process/governance
git commit -am "Advance governance pin"
```

A stale pin is not a defect. It records the governance this repo was working
under. Advance it when the newer guidance matters, and say so in the commit.

## Not included

Parent-organization policy is internal and lives in the private operations
repo. It is not required to read or apply the governance above, and is
deliberately not a submodule here — a private submodule would break cloning for
anyone outside the org.
