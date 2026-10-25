"""
Administrator-only account management (run on the server; never exposed in the app).

  python -m scripts.manage_users create-coordinator --email a@b.org --name "Dr A"
  python -m scripts.manage_users promote --email a@b.org
  python -m scripts.manage_users reset-password --email a@b.org

Passwords are prompted for (not accepted on the command line, so they do not
land in shell history). The database is data/ctai.db unless CTAI_DB_PATH is set.
"""

from __future__ import annotations

import argparse
import getpass
import sys

from services import auth


def _password() -> str:
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Confirm password: "):
        raise auth.AuthError("Passwords do not match.")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-coordinator")
    create.add_argument("--email", required=True)
    create.add_argument("--name", required=True)
    for name in ("promote", "reset-password"):
        sub.add_parser(name).add_argument("--email", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "create-coordinator":
            auth.create_user(args.name, args.email, _password(), auth.COORDINATOR)
            print(f"Coordinator account created for {auth.normalize_email(args.email)}.")
        elif args.command == "promote":
            auth.set_role(args.email, auth.COORDINATOR)
            print("Account promoted to coordinator; existing sessions were ended.")
        else:
            auth.admin_set_password(args.email, _password())
            print("Password updated; existing sessions were ended.")
    except auth.AuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
