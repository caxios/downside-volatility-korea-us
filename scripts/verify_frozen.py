"""
Verify the frozen rule files: recompute each SHA-256 hash from the stored rules and compare it with the
hash recorded at freezing time, then list test-period accesses in order.

  python scripts/verify_frozen.py
"""
import _bootstrap  # noqa: F401
from src.config import LOG_DIR
from src.firewall import FROZEN_DIR, FirewallError, load_frozen


def main():
    print(f"{'rule set':8s} {'frozen at':20s} {'sha256 (first 12)':18s} check")
    for path in sorted(FROZEN_DIR.glob("*.json")):
        name = path.stem
        try:
            p = load_frozen(name)          # raises if the stored rules no longer match the stored hash
            print(f"{name:8s} {p['frozen_at']:20s} {p['sha256'][:12]:18s} OK")
        except FirewallError as e:
            print(f"{name:8s} {'':20s} {'':18s} FAILED: {e}")
    log = LOG_DIR / "test_access.log"
    if log.exists():
        lines = log.read_text(encoding="utf-8").splitlines()
        print(f"\ntest-period accesses logged: {len(lines)} (first {lines[0].split(chr(9))[0]}, "
              f"last {lines[-1].split(chr(9))[0]})")


if __name__ == "__main__":
    main()
