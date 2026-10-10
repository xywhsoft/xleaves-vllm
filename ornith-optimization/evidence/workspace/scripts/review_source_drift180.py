"""Read-only comparison of current dirty source with recorded case/profile inputs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(repo, manifest):
    changed = set(subprocess.check_output(
        ['git', 'ls-files', '--modified', '--others', '--exclude-standard'],
        cwd=repo, text=True).splitlines())
    changed = {rel for rel in changed if (repo / rel).is_file()}
    pinned = {str(Path(item['path']).relative_to(repo)): item['sha256']
              for item in manifest if Path(item['path']).is_absolute()
              and Path(item['path']).is_relative_to(repo)}
    unbound = sorted(changed - set(pinned))
    mismatched = sorted(rel for rel in changed & set(pinned)
                        if sha(repo / rel) != pinned[rel])
    return dict(changed_files=len(changed), unbound_source=unbound,
                mismatched_source=mismatched,
                source_matches=not unbound and not mismatched)


def main():
    prior = json.loads((ROOT / 'results/qualified-parent166-profile156.json').read_text())
    repo = Path(prior['cwd'])
    parent = compare(repo, prior['file_manifest'])
    assert not parent['source_matches']
    assert 'flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py' in parent['unbound_source']
    assert 'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py' in parent['unbound_source']
    path = ROOT / 'results/strided_gdn177_capacity179.jsonl'
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    start = [r for r in rows if r['kind'] == 'case_start'][-1]
    manifest = [{**item, 'path': str(Path(start['qualification_spec']['cwd']) / item['path'])}
                for item in start['source_manifest']]
    current = compare(Path(start['qualification_spec']['cwd']), manifest)
    assert current['source_matches']
    completed = any(r['kind'] == 'case_complete' and r['case'] == start['case'] for r in rows)
    proof = dict(status='source_drift180_read_only_review_passed',
                 parent166_current_inputs=parent, live179_inputs=current,
                 live179_case=start['case'], live179_whole_model_qualified=completed,
                 deployment_performed=False,
                 implication='166 fallback requires restoring both original tracked files; matching179inputs alone does not qualify it for deployment.')
    output = ROOT / 'results/source-drift180.json'
    assert not output.exists()
    output.write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
