import pytest
from unittest.mock import patch, MagicMock, call

from src.shared import gcs_utils


@patch('gcsfs.GCSFileSystem')
def test_keep_only_latest_version_removes_old(mock_fs_class):
    fs = MagicMock()
    mock_fs_class.return_value = fs
    fs.exists.return_value = True
    fs.ls.return_value = [
        'bucket/prefix/20240101000000',
        'bucket/prefix/20240102000000',
        'bucket/prefix/20240103000000',
    ]
    fs.isdir.return_value = True

    gcs_utils.keep_only_latest_version('gs://bucket/prefix')

    expected_calls = [
        call('bucket/prefix/20240102000000', recursive=True),
        call('bucket/prefix/20240101000000', recursive=True),
    ]
    assert fs.rm.call_args_list == expected_calls
    assert not any('20240103000000' in c.args[0] for c in fs.rm.call_args_list)


@patch('gcsfs.GCSFileSystem')
def test_keep_only_latest_version_handles_malformed_uri(mock_fs_class):
    fs = MagicMock()
    mock_fs_class.return_value = fs
    fs.exists.return_value = True
    fs.ls.return_value = ['bucket/prefix/20240101000000', 'bucket/prefix/20240102000000']
    fs.isdir.return_value = True

    gcs_utils.keep_only_latest_version('gs:/bucket/prefix')

    fs.ls.assert_called_once_with('bucket/prefix', detail=False)
    assert fs.rm.call_count == 1


@patch('gcsfs.GCSFileSystem')
def test_keep_only_latest_version_nonexistent_prefix(mock_fs_class):
    fs = MagicMock()
    mock_fs_class.return_value = fs
    fs.exists.return_value = False

    gcs_utils.keep_only_latest_version('gs://bucket/missing')

    fs.rm.assert_not_called()
