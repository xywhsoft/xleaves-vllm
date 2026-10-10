"""Record serving inputs without storing credentials or copying model weights."""
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def effective_environment(environment):
    exact = {'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'CUDA_HOME',
             'CUDA_VISIBLE_DEVICES', 'TOKENIZERS_PARALLELISM',
             'OMP_NUM_THREADS', 'MKL_NUM_THREADS'}
    prefixes = ('VLLM_', 'NCCL_', 'GGML_', 'PYTORCH_')
    forbidden = ('API_KEY', 'PASSWORD', 'SECRET', 'ACCESS_TOKEN', 'AUTH_TOKEN')
    return {name: value for name, value in sorted(environment.items())
            if (name in exact or name.startswith(prefixes))
            and not any(word in name.upper() for word in forbidden)}


def artifact_manifest(case):
    repo = Path(case['cwd'])
    command = case['command']
    paths = set((repo / 'vllm').glob('*.so'))
    paths.add(repo / 'vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so')
    legacy = repo.parents[2] / 'src/onecat/flash-attention-v100/flash_attn_v100'
    paths.update(legacy.glob('*.so'))
    paths.update(Path(value) for name, value in case.get('env', {}).items()
                 if name.endswith('_LIBRARY'))
    for flag in ['--hf-config-path', '--tokenizer']:
        if flag in command:
            folder = Path(command[command.index(flag) + 1])
            paths.update(path for path in folder.rglob('*') if path.is_file())
    # The native manifest binds the binary to its exact sources and compiler flags.
    for path in list(paths):
        manifest = path.parent.parent / 'ornith-native-manifest.json'
        if manifest.is_file():
            paths.add(manifest)
    result = []
    for path in sorted(paths):
        if path.is_file():
            result.append({'path': str(path), 'bytes': path.stat().st_size,
                           'sha256': sha(path)})
        elif path.name.endswith('.so'):
            if str(path) in case.get('env', {}).values():
                raise FileNotFoundError(path)
    return result


def model_identity(command):
    result = []
    model = Path(command[command.index('serve') + 1])
    paths = {model}
    if '--hf-config-path' in command:
        folder = Path(command[command.index('--hf-config-path') + 1])
        config = json.loads((folder / 'config.json').read_text())
        if config.get('gguf_mmproj_path'):
            paths.add(Path(config['gguf_mmproj_path']))
    for path in sorted(paths):
        stat = path.stat()
        # Full weight hashes are intentionally a separate disk job, outside timing.
        result.append({'path': str(path), 'bytes': stat.st_size,
                       'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino,
                       'device': stat.st_dev})
    return result
