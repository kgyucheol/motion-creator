"""Batch generation entry point for future task schedulers."""
import argparse
import json
from pathlib import Path
from .robot import Robot
from .demo import crouch_demo
from .motion import save_bundle


def main():
    parser = argparse.ArgumentParser(description='G1 motion generation and export')
    subs = parser.add_subparsers(dest='command', required=True)
    demo = subs.add_parser('demo', help='Generate a grounded crouch/stand reference')
    demo.add_argument('--depth', type=float, default=.16, help='Pelvis lowering distance in metres')
    demo.add_argument('--fps', type=int, default=30)
    export = subs.add_parser('export', help='Export a saved editable JSON project')
    export.add_argument('project', type=Path)
    export.add_argument('--fps', type=int, default=30)
    demo.add_argument('--protomotions', action='store_true', help='Also export native ProtoMotions .motion/.pt')
    export.add_argument('--protomotions', action='store_true', help='Also export native ProtoMotions .motion/.pt')
    convert = subs.add_parser('protomotions', help='Convert an existing NPZ without changing the source')
    convert.add_argument('reference', type=Path)
    args = parser.parse_args()
    if args.command == 'protomotions':
        from .protomotions_bridge import export_isolated
        print(json.dumps({'files': export_isolated(args.reference.resolve())}, ensure_ascii=False, indent=2))
        return
    robot = Robot()
    project = crouch_demo(robot, args.depth) if args.command == 'demo' else json.loads(args.project.read_text())
    print(json.dumps(save_bundle(robot, project, args.fps, protomotions=args.protomotions), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
