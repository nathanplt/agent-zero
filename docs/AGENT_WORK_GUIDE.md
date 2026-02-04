# Agent Work Guide

## When You Start

1. Read `STATUS.md` - find an `available` feature
2. Read that feature's spec in `PROJECT_PLAN.md`
3. Check dependencies are complete (in STATUS.md Completed section)
4. Claim it: update STATUS.md with `in progress` and your ID
5. Create branch: `feature/X.Y-name`

## How to Work

1. Write tests first (they should fail)
2. Implement until tests pass
3. Run: `make test && make lint && make typecheck`
4. Commit and report done

## When Done

1. Update STATUS.md: move feature to Completed
2. Add next available features from PROJECT_PLAN.md to Current Work
3. Tell the user you're done

## Rules

- One feature at a time
- Don't start if dependencies aren't complete
- Tests before implementation
- Don't modify interfaces without coordination
