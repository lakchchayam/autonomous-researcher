"""
CLI entrypoint for the Autonomous Researcher.

Usage:
    python -m src.cli "Research the differences between Claude 3.5, GPT-4o, and Gemini 1.5 Pro"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .agent import research


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous AI Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.cli "Compare LangGraph vs CrewAI for multi-agent orchestration"
    python -m src.cli --max-iterations 5 --output report.md "Analyze 2024 AI agent frameworks"
        """,
    )
    
    parser.add_argument(
        "goal",
        type=str,
        help="The research goal or question to investigate",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum research-critique cycles (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the report (default: stdout)",
    )
    
    args = parser.parse_args()
    
    print(f"\n🔬 Autonomous Researcher v0.1.4")
    print(f"📋 Goal: {args.goal}")
    print(f"🔄 Max iterations: {args.max_iterations}")
    print(f"{'=' * 60}\n")
    
    report = asyncio.run(research(args.goal, max_iterations=args.max_iterations))
    
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\n✅ Report saved to: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
