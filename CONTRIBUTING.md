# Contributing

Thanks for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

```bash
git clone https://github.com/<org>/splat-web-viewer.git
cd splat-web-viewer
npm install
```

Run the test suite:

```bash
npm test
```

Start the dev server:

```bash
npm run dev
```

## Code Style

This project uses TypeScript. Ensure your code compiles without errors:

```bash
npm run typecheck
npm run lint
```

Follow the existing code conventions. Use `prettier` for formatting if configured, otherwise match the style of surrounding code.

## Submitting a Pull Request

1. Fork the repository and clone your fork.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b my-feature
   ```
3. Make your changes, add tests where appropriate.
4. Ensure tests pass and the build succeeds.
5. Push to your fork and open a pull request against `main`.

### PR Guidelines

- Keep PRs focused on a single change.
- Write a clear description of what the PR does and why.
- Link any related issues.
- Ensure CI passes before requesting review.

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
