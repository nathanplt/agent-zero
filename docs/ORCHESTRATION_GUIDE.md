# Orchestration Guide

## Your Workflow

1. **Start new chat** (Cmd+L)
2. **Say**: "Read PROJECT_PLAN.md and AGENT_WORK_GUIDE.md. Work on the next available feature."
3. **Agent does the work**, updates STATUS.md when done
4. **Repeat** with more chats for parallel work

That's it. Agents self-coordinate via STATUS.md.

## If Something Breaks

- **Merge conflict**: "Pull latest and rebase"
- **Agent confused**: "Focus only on Feature X.Y"
- **Tests fail**: "Fix the failing tests"

## Max Parallel: 3-4 agents
