# Git Push Error: Large Files Fix

## Problem

Cannot push to GitHub because commit `ea3684b` contains three large .svs files (126-186 MB) that exceed GitHub's 100 MB limit:
- `images/129753.svs` (127 MB)
- `images/129754.svs` (174 MB)
- `images/129755.svs` (186 MB)

Error message:
```
remote: error: File images/129753.svs is 126.16 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage
```

## Solution Options

### Option 1: Remove Large Files from Git History (Recommended)

Use BFG Repo-Cleaner to remove the large files from all commits:

```bash
# Download BFG Repo-Cleaner
cd /tmp
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# Clone a fresh copy of your repo
cd ~
git clone --mirror https://github.com/smith6jt-cop/isletscope.git isletscope-clean.git

# Remove large files from history
java -jar /tmp/bfg-1.14.0.jar --delete-files '*.svs' isletscope-clean.git

# Clean up
cd isletscope-clean.git
git reflog expire --expire=now --all && git gc --prune=now --aggressive

# Force push (WARNING: This rewrites history)
git push --force
```

### Option 2: Use Git Filter-Branch (Manual)

```bash
cd /home/smith6jt/isletscope

# Remove large files from all commits
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch 'images/*.svs'" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up refs
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push
git push origin main --force
```

### Option 3: Reset and Recommit (Easiest)

Since you haven't pushed these commits yet, just reset and recommit without the large files:

```bash
cd /home/smith6jt/isletscope

# Reset to the first commit (before large files were added)
git reset --soft ba6ecf7

# The .gitignore is already created and will prevent SVS files from being added
git status  # Should show all changes staged

# Remove any SVS files from staging
git reset HEAD images/*.svs

# Commit everything except large files
git commit -m "Complete IsletScope implementation with InstanSeg integration

- Automated stain normalization (Macenko/Vahadane)
- Cell segmentation with InstanSeg (fixed shape broadcasting errors)
- Tile-based processing for large WSI images
- Islet detection and radial analysis
- Tissue classification and 3D spatial inference
- Complete notebook workflow (00-03)
- Training notebook for custom InstanSeg models
- Comprehensive documentation (QUICKSTART.md, INSTANSEG_FIX.md)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Now push to GitHub
git push origin main
```

## Current Status

- ✅ `.gitignore` created (excludes `*.svs` files)
- ✅ Large files removed from working directory index
- ⏳ Large files still exist in commit `ea3684b` history
- ⏳ Need to clean history before pushing

## Recommended Action

I recommend **Option 3 (Reset and Recommit)** because:
1. ✅ You haven't pushed these commits yet (no one else affected)
2. ✅ Simpler than filter-branch
3. ✅ Creates clean commit history
4. ✅ `.gitignore` already prevents future large file commits

## After Fixing

Once history is clean:

```bash
# Verify no large files in history
git log --all --format='%H' --reverse | while read commit; do
  size=$(git cat-file -s $commit 2>/dev/null || echo 0)
  if [ $size -gt 100000000 ]; then
    echo "Large file in $commit"
    git ls-tree -r $commit | awk '{if ($4 > 100000000) print $4, $5}'
  fi
done

# Push to GitHub
git push origin main

# Verify push succeeded
git log --oneline origin/main
```

## Keeping Image Files Locally

The `.svs` files are still in your `images/` directory (not deleted, just untracked by git). They will remain on your local machine for running notebooks.

To share images with collaborators:
1. Upload to a cloud storage service (Google Drive, Dropbox, Box, etc.)
2. Or use Git LFS (Large File Storage) - requires setup
3. Or document where to obtain the images in README

## Summary

**Current situation**:
- Local commits have large files → cannot push to GitHub
- `.gitignore` created → future commits won't include large files

**Next step**:
- Choose Option 3 (reset and recommit) to create clean history
- Then push to GitHub successfully
