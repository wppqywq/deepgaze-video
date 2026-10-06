# Publishing a source snapshot

Use a fresh Git history for the first public release. The older workspace history contains retired internal records and large presentation files. Deleting them from the current tree does not remove them from past commits. Keep the existing repository and remote unchanged.

After the reviewed source changes have been committed, export only tracked files to a new, empty directory:

```bash
git archive HEAD | tar -x -C /path/to/empty-public-directory
cd /path/to/empty-public-directory
git init -b main
git add .
git commit -m "Publish video gaze adaptation source and development results"
```

Check the file list before publication. Include code, tests, English documentation, aggregate results, and the plot. Exclude datasets, original trial provenance, model weights, run logs, environments, caches, credentials, and backups. The current ignore rules support this separation. No project license is supplied, as requested; third-party resources retain their own terms.

Create an empty public repository at [GitHub](https://github.com/new). Do not initialize it with a README, license, or ignore file. Authenticate with your own GitHub account and add the new remote in the snapshot directory:

```bash
git remote add origin git@github.com:YOUR_USERNAME/deepgaze-video.git
git push -u origin main
```

Replace `YOUR_USERNAME` and the repository name with your choices. GitHub also supports an HTTPS remote. These steps follow its [existing-source publication guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

The public source and summary results do not constitute a complete reproducibility package: project-specific trial inputs and trained adapters are not distributed. The README and manuscript state this limitation.
