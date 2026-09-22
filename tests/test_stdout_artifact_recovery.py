import pytest
from cascadekv import stdout_artifact_recovery as recovery

def test_base64_stdout_round_trip_is_exact_and_framed(tmp_path):
    source = tmp_path / "development.json"; source.write_bytes(b'{"no_raw_text":true}\n')
    stream = recovery.emit({"development.json": source})
    assert "BEGIN CASCADEKV C5 RECOVERY MANIFEST" in stream
    restored = recovery.recover(stream, tmp_path / "restored")
    assert restored["development.json"].read_bytes() == source.read_bytes()
    assert recovery.sha256_bytes(restored["development.json"].read_bytes()) == recovery.sha256_bytes(source.read_bytes())

@pytest.mark.parametrize("mutation",[
    lambda s:s.replace('"base64":"','"base64":"%',1),
    lambda s:s.replace('"byte_count":21','"byte_count":22',1),
    lambda s:s.replace('"sha256":"','"sha256":"0',1),
    lambda s:s.replace("development.json =====","other.json =====",1),
    lambda s:s+"===== BEGIN CASCADEKV C5 ARTIFACT extra.json =====\n{}\n===== END CASCADEKV C5 ARTIFACT extra.json =====\n",
    lambda s:s.replace("===== END CASCADEKV C5 ARTIFACT development.json =====","",1),
    lambda s:s.replace("development.json","../development.json",1),
])
def test_recovery_rejects_corruption(tmp_path, mutation):
    source=tmp_path/"development.json"; source.write_bytes(b'{"no_raw_text":true}\n')
    with pytest.raises(recovery.RecoveryError): recovery.recover(mutation(recovery.emit({"development.json":source})),tmp_path/"out")

def test_recovery_multi_artifact_round_trip(tmp_path):
    first,second=tmp_path/"a.json",tmp_path/"b.json"; first.write_bytes(b"a"); second.write_bytes(b"b\x00")
    restored=recovery.recover(recovery.emit({"b.json":second,"a.json":first}),tmp_path/"out")
    assert {name:path.read_bytes() for name,path in restored.items()} == {"a.json":b"a","b.json":b"b\x00"}

def test_recovery_binary_zero_byte_and_exact_hashes(tmp_path):
    zero,binary=tmp_path/"zero",tmp_path/"binary"; zero.write_bytes(b""); binary.write_bytes(b"\x00\xff\x80not-utf8\n")
    stream=recovery.emit({"zero":zero,"binary":binary}); restored=recovery.recover(stream,tmp_path/"out")
    assert {n:p.read_bytes() for n,p in restored.items()}=={"zero":b"","binary":binary.read_bytes()}
    assert all(recovery.sha256_bytes(p.read_bytes())==recovery.sha256_bytes(({"zero":zero,"binary":binary})[n].read_bytes()) for n,p in restored.items())

@pytest.mark.parametrize("mutation",[
    lambda s:s.replace('"base64":"','"base64":"%',1), # malformed base64
    lambda s:s.replace('"sha256":"','"sha256":"f',1), # valid framing, bad byte hash
    lambda s:s.replace('"byte_count":21','"byte_count":20',1),
    lambda s:s.replace('"sha256":"','"sha256":"z',1), # malformed manifest SHA
    lambda s:s.replace('"schema_version":"cascadekv-c5-stdout-recovery-v1"','"schema_version":"wrong"'),
    lambda s:s.replace('development.json =====','other.json =====',1), # block filename mismatch
    lambda s:s.replace('===== BEGIN CASCADEKV C5 RECOVERY MANIFEST =====','===== BEGIN CASCADEKV C5 RECOVERY MANIFEST =====\n{}\n===== END CASCADEKV C5 RECOVERY MANIFEST =====\n===== BEGIN CASCADEKV C5 RECOVERY MANIFEST =====',1),
    lambda s:s.replace('===== END CASCADEKV C5 RECOVERY MANIFEST =====','===== END CASCADEKV C5 RECOVERY MANIFEST =====\n===== END CASCADEKV C5 RECOVERY MANIFEST =====',1),
    lambda s:s.replace('===== BEGIN CASCADEKV C5 ARTIFACT development.json =====','===== BEGIN CASCADEKV C5 ARTIFACT development.json =====\n===== BEGIN CASCADEKV C5 ARTIFACT development.json =====',1),
    lambda s:s.replace('===== END CASCADEKV C5 ARTIFACT development.json =====','===== END CASCADEKV C5 ARTIFACT development.json =====\n===== END CASCADEKV C5 ARTIFACT development.json =====',1),
    lambda s:s.replace('===== BEGIN CASCADEKV C5 RECOVERY MANIFEST =====','x',1), lambda s:s.replace('===== END CASCADEKV C5 RECOVERY MANIFEST =====','x',1),
    lambda s:s.replace('===== BEGIN CASCADEKV C5 ARTIFACT development.json =====','x',1), lambda s:s.replace('===== END CASCADEKV C5 ARTIFACT development.json =====','x',1),
    lambda s:s.replace('"artifacts":[{','"artifacts":[{"filename":"missing","byte_count":0,"sha256":"'+"0"*64+'"},{',1),
    lambda s:s+'===== BEGIN CASCADEKV C5 ARTIFACT extra =====\n{"base64":"","byte_count":0,"filename":"extra","sha256":"'+"0"*64+'"}\n===== END CASCADEKV C5 ARTIFACT extra =====\n',
    lambda s:s.replace('development.json','../x',1), lambda s:s.replace('development.json','/absolute',1), lambda s:s.replace('development.json','nested/name',1),
    lambda s:s.replace('{"artifacts"','{bad',1), lambda s:s.replace('{"base64"','{bad',1),
    lambda s:s+'===== END CASCADEKV C5 ARTIFACT ghost =====\n',
])
def test_recovery_complete_corruption_matrix(tmp_path,mutation):
    source=tmp_path/"development.json"; source.write_bytes(b'{"no_raw_text":true}\n')
    with pytest.raises(recovery.RecoveryError): recovery.recover(mutation(recovery.emit({"development.json":source})),tmp_path/"out")

def test_recovery_rejects_duplicate_manifest_filename_and_metadata_disagreement(tmp_path):
    first,second=tmp_path/"a",tmp_path/"b"; first.write_bytes(b"a"); second.write_bytes(b"b")
    stream=recovery.emit({"a":first,"b":second})
    # Duplicate declaration, block SHA mismatch, block count mismatch, and reordered framing all fail closed.
    cases=[stream.replace('"filename":"b"','"filename":"a"',1),stream.replace('"sha256":"'+recovery.sha256_bytes(b"a")+'"','"sha256":"'+"0"*64+'"',2),stream.replace('"byte_count":1,"filename":"a"','"byte_count":2,"filename":"a"',1),stream.replace('===== BEGIN CASCADEKV C5 ARTIFACT a =====','===== BEGIN CASCADEKV C5 ARTIFACT b =====',1)]
    for bad in cases:
        with pytest.raises(recovery.RecoveryError): recovery.recover(bad,tmp_path/"out")
