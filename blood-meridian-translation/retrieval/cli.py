"""CLI entry point for the retrieval system.

Usage:
    python3 -m retrieval harvest          # harvest corpus from Perseus/Scaife
    python3 -m retrieval embed            # build FAISS indices at all scales
    python3 -m retrieval tag              # build construction-type tags
    python3 -m retrieval collocates       # build collocate PMI index
    python3 -m retrieval search "query"   # search with options
    python3 -m retrieval search -i        # interactive search mode
    python3 -m retrieval status           # show index sizes and status
"""

from __future__ import annotations

import argparse
import logging
import sys

from .schemas import Scale, ConstructionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_retrieval_result(result) -> str:
    """Format a single RetrievalResult for terminal display."""
    c = result.chunk
    lines = [
        f"  [{result.rank}] score={result.score:.4f}  {c.author} ({c.period}) — {c.work} {getattr(c, 'record_id', '')}",
        f"      {c.text[:200]}{'...' if len(c.text) > 200 else ''}",
    ]
    return "\n".join(lines)


def _format_collocate(entry, rank: int) -> str:
    """Format a single CollocateEntry for terminal display."""
    return f"  [{rank}] {entry.collocate:20s}  PMI={entry.pmi:+.3f}  freq={entry.frequency}"


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_harvest(args: argparse.Namespace) -> None:
    """Run corpus harvest pipeline."""
    from .harvest import harvest_all
    print("Harvesting corpus...")
    n = harvest_all()
    print(f"Harvested {n} records.")


def _cmd_embed(args: argparse.Namespace) -> None:
    """Build FAISS indices at all scales."""
    from .embed import build_all_indices
    force = getattr(args, "force", False)
    print("Building embedding indices...")
    stats = build_all_indices(force_rebuild=force)
    print("\nIndex Build Summary:")
    for scale_name, n in stats.items():
        print(f"  {scale_name:10s}: {n:>8,d} vectors")
    print(f"  {'TOTAL':10s}: {sum(stats.values()):>8,d} vectors")


def _cmd_tag(args: argparse.Namespace) -> None:
    """Build construction-type tags for the corpus."""
    from .search import build_construction_tags
    print("Tagging corpus with construction types...")
    n = build_construction_tags()
    print(f"Tagged {n} records.")


def _cmd_collocates(args: argparse.Namespace) -> None:
    """Build the collocate PMI index."""
    from .collocate_index import build_collocate_index
    print("Building collocate index...")
    index = build_collocate_index()
    print(f"Built collocate index: {len(index)} lemmata with collocates.")


def _cmd_search(args: argparse.Namespace) -> None:
    """Run a search query."""
    from .search import search

    # Interactive mode
    if getattr(args, "interactive", False):
        _interactive_search()
        return

    query = args.query
    if not query:
        print("Error: provide a query string or use -i for interactive mode.", file=sys.stderr)
        sys.exit(1)

    mode = args.mode or "lexical"
    scale = Scale(args.scale) if args.scale else Scale.SENTENCE
    top_k = args.top_k or 10
    period = args.period or None

    construction_type = None
    if args.construction:
        construction_type = ConstructionType(args.construction)

    print(f"Searching: mode={mode}, scale={scale.value}, top_k={top_k}")
    if period:
        print(f"  period filter: {period}")
    if construction_type:
        print(f"  construction filter: {construction_type.value}")
    print()

    results = search(
        query=query,
        mode=mode,
        scale=scale,
        construction_type=construction_type,
        period_filter=period,
        top_k=top_k,
    )

    if not results:
        print("No results found.")
        return

    if mode == "collocate":
        print(f"Collocates for '{query}':")
        for i, entry in enumerate(results, 1):
            print(_format_collocate(entry, i))
    else:
        print(f"Results ({len(results)}):")
        for result in results:
            print(_format_retrieval_result(result))
            print()


def _cmd_status(args: argparse.Namespace) -> None:
    """Show index status and sizes."""
    from .search import index_status

    status = index_status()
    print("Retrieval System Status")
    print("=" * 60)

    for name, info in status.items():
        exists = info.get("exists", False)
        marker = "OK" if exists else "--"
        print(f"\n  [{marker}] {name}")
        if exists:
            for k, v in info.items():
                if k == "exists":
                    continue
                print(f"       {k}: {v}")
        else:
            print("       Not built yet.")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def _interactive_search() -> None:
    """Interactive search REPL."""
    from .search import search

    print("Interactive Retrieval Search")
    print("Commands: /mode <lexical|syntactic|register|collocate>")
    print("          /scale <phrase|sentence|passage>")
    print("          /period <homeric|archaic|classical|hellenistic|koine|imperial>")
    print("          /construction <paratactic_narrative|...>")
    print("          /top <N>")
    print("          /quit")
    print()

    mode = "lexical"
    scale = Scale.SENTENCE
    top_k = 10
    period_filter = None
    construction_type = None

    while True:
        try:
            line = input(f"[{mode}/{scale.value}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        # Handle slash commands
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            val = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/q"):
                print("Bye.")
                break
            elif cmd == "/mode":
                if val in ("lexical", "syntactic", "register", "collocate"):
                    mode = val
                    print(f"  Mode: {mode}")
                else:
                    print(f"  Unknown mode: {val}. Options: lexical, syntactic, register, collocate")
            elif cmd == "/scale":
                try:
                    scale = Scale(val)
                    print(f"  Scale: {scale.value}")
                except ValueError:
                    print(f"  Unknown scale: {val}. Options: phrase, sentence, passage")
            elif cmd == "/period":
                if val:
                    period_filter = val
                    print(f"  Period filter: {period_filter}")
                else:
                    period_filter = None
                    print("  Period filter: cleared")
            elif cmd == "/construction":
                if val:
                    try:
                        construction_type = ConstructionType(val)
                        print(f"  Construction: {construction_type.value}")
                    except ValueError:
                        print(f"  Unknown construction: {val}")
                        print(f"  Options: {', '.join(ct.value for ct in ConstructionType)}")
                else:
                    construction_type = None
                    print("  Construction filter: cleared")
            elif cmd == "/top":
                try:
                    top_k = int(val)
                    print(f"  Top K: {top_k}")
                except ValueError:
                    print(f"  Invalid number: {val}")
            else:
                print(f"  Unknown command: {cmd}")
            continue

        # Run search
        try:
            results = search(
                query=line,
                mode=mode,
                scale=scale,
                construction_type=construction_type,
                period_filter=period_filter,
                top_k=top_k,
            )

            if not results:
                print("  No results.")
            elif mode == "collocate":
                for i, entry in enumerate(results, 1):
                    print(_format_collocate(entry, i))
            else:
                for result in results:
                    print(_format_retrieval_result(result))
                    print()
        except FileNotFoundError as e:
            print(f"  Error: {e}")
        except Exception as e:
            print(f"  Error: {e}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrieval",
        description="Blood Meridian translation retrieval system",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # harvest
    subparsers.add_parser("harvest", help="Harvest corpus from Perseus/Scaife")

    # embed
    sub_embed = subparsers.add_parser("embed", help="Build FAISS embedding indices")
    sub_embed.add_argument("--force", action="store_true", help="Force rebuild existing indices")

    # tag
    subparsers.add_parser("tag", help="Build construction-type tags for corpus")

    # collocates
    subparsers.add_parser("collocates", help="Build collocate PMI index")

    # search
    sub_search = subparsers.add_parser("search", help="Search the corpus")
    sub_search.add_argument("query", nargs="?", default=None, help="Search query")
    sub_search.add_argument(
        "--mode", "-m",
        choices=["lexical", "syntactic", "register", "collocate"],
        default="lexical",
        help="Search mode (default: lexical)",
    )
    sub_search.add_argument(
        "--scale", "-s",
        choices=[s.value for s in Scale],
        default="sentence",
        help="Embedding scale (default: sentence)",
    )
    sub_search.add_argument(
        "--construction", "-c",
        choices=[ct.value for ct in ConstructionType],
        default=None,
        help="Construction type filter (for syntactic mode)",
    )
    sub_search.add_argument(
        "--period", "-p",
        default=None,
        help="Period filter (e.g. classical, koine, hellenistic)",
    )
    sub_search.add_argument(
        "--top-k", "-k",
        type=int, default=10,
        help="Number of results (default: 10)",
    )
    sub_search.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Enter interactive search mode",
    )

    # status
    subparsers.add_parser("status", help="Show index status and sizes")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "harvest": _cmd_harvest,
        "embed": _cmd_embed,
        "tag": _cmd_tag,
        "collocates": _cmd_collocates,
        "search": _cmd_search,
        "status": _cmd_status,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)
