# Commit Message Guidelines

## Format
<type>(<scope>): <description>

## Types
- feat: A new feature
- fix: A bug fix
- docs: Documentation changes
- style: Formatting changes (no code changes)
- refactor: Code changes that neither fix a bug nor add a feature
- perf: A code change that improves performance
- test: Adding or updating tests
- chore: Changes to the build process or auxiliary tools and libraries

## Examples
feat(api): add user authentication endpoint
fix(ui): correct button alignment on login page
docs: update README with installation instructions

## Additional Guidelines
- **Keep Descriptions Concise**: Aim for clarity, limit the summary to ~50 characters.
- **Group Related Changes**: Batch related changes into a single commit.
- **Use Special Keywords**:
  - WIP: Work In Progress for ongoing changes.
  - BREAKING CHANGE: Highlight changes that break backward compatibility.

## Example with Additional Details
fix(database): resolve connection timeout error

This commit fixes the issue where database connections would time out 
after 30 seconds. Adjusted connection settings to ensure stability.
