# Publishing Guide for langgraph-responses-gateway

This guide walks you through publishing the package to PyPI (Python Package Index).

## Prerequisites

### 1. Create PyPI Account
1. Go to https://pypi.org/account/register/
2. Create an account with a strong password
3. Verify your email address
4. Enable 2FA (Two-Factor Authentication) for security

### 2. Create API Token
1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens" section
3. Click "Add API token"
4. Name: `langgraph-responses-gateway`
5. Scope: "Entire account" (for first publish) or specific to this project later
6. Copy the token (starts with `pypi-`)
7. **Save it securely** - you won't see it again!

### 3. Configure Token Locally
```bash
# Create .pypirc file in your home directory
cat > ~/.pypirc << EOF
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = YOUR_PYPI_TOKEN_HERE
EOF

# Set proper permissions
chmod 600 ~/.pypirc
```

## Building the Package

### 1. Install Build Tools
```bash
# Using uv (recommended)
uv pip install build twine

# Or using pip
pip install build twine
```

### 2. Clean Previous Builds
```bash
rm -rf dist/ build/ *.egg-info/
```

### 3. Build Distribution Files
```bash
# Build both wheel and source distribution
python -m build

# This creates:
# - dist/langgraph_responses_gateway-0.1.0-py3-none-any.whl (wheel)
# - dist/langgraph_responses_gateway-0.1.0.tar.gz (source)
```

### 4. Verify the Build
```bash
# Check the distribution files
twine check dist/*

# You should see:
# Checking dist/langgraph_responses_gateway-0.1.0-py3-none-any.whl: PASSED
# Checking dist/langgraph_responses_gateway-0.1.0.tar.gz: PASSED
```

## Publishing to PyPI

### Option 1: Test on TestPyPI First (Recommended)
```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ langgraph-responses-gateway

# If everything works, proceed to real PyPI
```

### Option 2: Direct to PyPI
```bash
# Upload to PyPI
twine upload dist/*

# You'll see output like:
# Uploading distributions to https://upload.pypi.org/legacy/
# Uploading langgraph_responses_gateway-0.1.0-py3-none-any.whl
# Uploading langgraph_responses_gateway-0.1.0.tar.gz
# 
# View at:
# https://pypi.org/project/langgraph-responses-gateway/0.1.0/
```

## Post-Publication

### 1. Test Installation
```bash
# In a fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from PyPI
pip install langgraph-responses-gateway

# Test import
python -c "from langgraph_responses_gateway import ResponsesGateway; print('Success!')"
```

### 2. Create GitHub Release
```bash
# Tag the version
git tag -a v0.1.0 -m "Release v0.1.0 - Initial release"
git push origin v0.1.0

# Create release on GitHub
gh release create v0.1.0 \
  --title "v0.1.0 - Initial Release" \
  --notes "Initial release of langgraph-responses-gateway

## Features
- Full OpenAI Responses API compliance
- Streaming SSE support
- Zero configuration setup
- Thread and user isolation support

## Installation
\`\`\`bash
pip install langgraph-responses-gateway
\`\`\`

See README for usage examples."
```

### 3. Update README Badge
Add to your README.md:
```markdown
[![PyPI version](https://badge.fury.io/py/langgraph-responses-gateway.svg)](https://badge.fury.io/py/langgraph-responses-gateway)
[![Downloads](https://pepy.tech/badge/langgraph-responses-gateway)](https://pepy.tech/project/langgraph-responses-gateway)
```

## Version Management

### Updating Version
1. Update version in `src/langgraph_responses_gateway/version.py`
2. Update version in `pyproject.toml`
3. Commit changes: `git commit -am "chore: bump version to X.Y.Z"`
4. Tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
5. Push: `git push && git push --tags`
6. Build and publish as above

### Version Numbering (Semantic Versioning)
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

## Automation with GitHub Actions (Optional)

Create `.github/workflows/publish.yml`:
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Then add your PyPI token as a GitHub secret:
1. Go to repo Settings → Secrets → Actions
2. Add new secret: `PYPI_API_TOKEN`
3. Value: Your PyPI token

## Troubleshooting

### Common Issues

1. **"The credential associated with user '__token__' isn't allowed to upload"**
   - Your token might be scoped wrong
   - Create a new token with "Entire account" scope

2. **"Package already exists"**
   - You can't overwrite existing versions
   - Bump the version number and republish

3. **"Invalid distribution file"**
   - Run `twine check dist/*` to validate
   - Rebuild with `python -m build`

### Getting Help
- PyPI documentation: https://packaging.python.org/
- Twine documentation: https://twine.readthedocs.io/
- Create issue: https://github.com/jero2rome/langgraph-responses-gateway/issues

## Checklist Before Publishing

- [ ] Version updated in `version.py` and `pyproject.toml`
- [ ] README.md is complete and accurate
- [ ] Examples work correctly
- [ ] Tests pass (if you have tests)
- [ ] LICENSE file is present
- [ ] .gitignore excludes build artifacts
- [ ] PyPI account created and verified
- [ ] API token generated and saved
- [ ] Test on TestPyPI first (optional but recommended)

## Quick Commands Summary

```bash
# One-time setup
pip install build twine
echo "[pypi]\nusername = __token__\npassword = YOUR_TOKEN" > ~/.pypirc

# For each release
rm -rf dist/ build/
python -m build
twine check dist/*
twine upload dist/*

# Verify
pip install langgraph-responses-gateway
```