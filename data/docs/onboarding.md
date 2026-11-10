# Getting Started with Aurora Cloud

## Regions

Aurora Cloud runs in five regions: us-east (Virginia), us-west (Oregon), eu-central (Frankfurt), ap-south (Mumbai) and ap-northeast (Tokyo). New projects default to us-east. A project's home region cannot be changed after creation, but resources can be deployed to any region from any project.

## The aurora CLI

Install the command line tool with `pip install aurora-cli`; it requires Python 3.10 or newer. Run `aurora login` to authenticate through the browser, then `aurora init` in your repository to create an `aurora.toml` configuration file. `aurora deploy` builds and ships the current directory, and `aurora rollback` restores the previous release. Each project keeps the last 10 releases available for rollback.

## Quotas

New accounts start with a quota of 32 vCPUs and 128 GB of memory per region, and 5 projects per organisation. Quota increases are requested from the console under Settings, then Quotas, and are usually approved within one business day. Requests above 256 vCPUs require a call with the solutions team.

## Support channels

Community support is on the forum at community.auroracloud.example. Business plan customers email support and Enterprise customers have a dedicated phone line. All plans can read the public documentation and the status page.
