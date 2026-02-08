"""
Tests for SLURM job management.

These tests verify SBATCH script generation and dry-run mode
without requiring an actual SLURM cluster.
"""

from cflibs.hpc import (
    ArrayJobConfig,
    SlurmJobConfig,
    SlurmJobManager,
    SlurmJobState,
    SlurmJobStatus,
)


def test_slurm_job_config_defaults():
    """Test default values for SlurmJobConfig."""
    config = SlurmJobConfig()
    assert config.job_name == "cflibs"
    assert config.partition == "default"
    assert config.nodes == 1
    assert config.ntasks == 1
    assert config.cpus_per_task == 1
    assert config.mem_gb == 4
    assert config.time_limit == "01:00:00"
    assert config.account is None
    assert config.output_path == "slurm-%j.out"
    assert config.error_path == "slurm-%j.err"
    assert config.extra_sbatch == {}
    assert config.env_vars == {}
    assert config.modules == []


def test_array_job_config():
    """Test ArrayJobConfig with array-specific fields."""
    config = ArrayJobConfig(
        job_name="test_array",
        array_size=100,
        max_concurrent=10,
    )
    assert config.job_name == "test_array"
    assert config.array_size == 100
    assert config.max_concurrent == 10


def test_generate_sbatch_basic():
    """Test basic SBATCH script generation."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(
        job_name="test_job",
        partition="compute",
        nodes=2,
        ntasks=4,
        cpus_per_task=8,
        mem_gb=16,
        time_limit="02:30:00",
    )

    script = manager.generate_sbatch_script(config, "echo 'Hello World'")

    # Check shebang
    assert script.startswith("#!/bin/bash")

    # Check standard directives
    assert "#SBATCH --job-name=test_job" in script
    assert "#SBATCH --partition=compute" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks=4" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=02:30:00" in script
    assert "#SBATCH --output=slurm-%j.out" in script
    assert "#SBATCH --error=slurm-%j.err" in script

    # Check script content
    assert "echo 'Hello World'" in script


def test_generate_sbatch_array():
    """Test SBATCH script generation with array job."""
    manager = SlurmJobManager(dry_run=True)
    config = ArrayJobConfig(
        job_name="test_array",
        array_size=100,
        max_concurrent=10,
    )

    script = manager.generate_sbatch_script(config, "python script.py $SLURM_ARRAY_TASK_ID")

    # Check array directive with limit
    assert "#SBATCH --array=0-99%10" in script
    assert "python script.py $SLURM_ARRAY_TASK_ID" in script


def test_generate_sbatch_array_unlimited():
    """Test SBATCH script with unlimited concurrent array tasks."""
    manager = SlurmJobManager(dry_run=True)
    config = ArrayJobConfig(
        job_name="test_array",
        array_size=50,
        max_concurrent=0,  # Unlimited
    )

    script = manager.generate_sbatch_script(config, "echo $SLURM_ARRAY_TASK_ID")

    # Check array directive without limit
    assert "#SBATCH --array=0-49" in script
    assert "%" not in script.split("#SBATCH --array=")[1].split("\n")[0]


def test_generate_sbatch_modules():
    """Test module load commands in SBATCH script."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(modules=["python/3.10", "cuda/11.8", "hdf5/1.12"])

    script = manager.generate_sbatch_script(config, "python script.py")

    assert "module load python/3.10" in script
    assert "module load cuda/11.8" in script
    assert "module load hdf5/1.12" in script


def test_generate_sbatch_env_vars():
    """Test environment variable export in SBATCH script."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(
        env_vars={
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "8",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
    )

    script = manager.generate_sbatch_script(config, "python script.py")

    assert "export JAX_PLATFORMS=cpu" in script
    assert "export OMP_NUM_THREADS=8" in script
    assert "export CUDA_VISIBLE_DEVICES=0,1" in script


def test_generate_sbatch_extra_directives():
    """Test extra SBATCH directives."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(
        extra_sbatch={
            "gres": "gpu:2",
            "constraint": "haswell",
            "mail-type": "END,FAIL",
            "mail-user": "user@example.com",
        }
    )

    script = manager.generate_sbatch_script(config, "python script.py")

    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --constraint=haswell" in script
    assert "#SBATCH --mail-type=END,FAIL" in script
    assert "#SBATCH --mail-user=user@example.com" in script


def test_generate_sbatch_account():
    """Test account directive when specified."""
    manager = SlurmJobManager(dry_run=True)

    # With account
    config_with = SlurmJobConfig(account="proj123")
    script_with = manager.generate_sbatch_script(config_with, "echo test")
    assert "#SBATCH --account=proj123" in script_with

    # Without account
    config_without = SlurmJobConfig(account=None)
    script_without = manager.generate_sbatch_script(config_without, "echo test")
    assert "--account" not in script_without


def test_dry_run_submit():
    """Test job submission in dry-run mode."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(job_name="test_dry_run")

    job_id = manager.submit(config, "echo 'test'")

    assert job_id == "DRY_RUN_test_dry_run"


def test_submit_with_dependency_dry_run():
    """Test dependency job submission in dry-run mode."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(job_name="dependent_job")

    # Submit with dependencies
    job_id = manager.submit_with_dependency(
        config,
        "echo 'depends on previous jobs'",
        depends_on=["12345", "67890"],
        dependency_type="afterok",
    )

    assert job_id == "DRY_RUN_dependent_job"

    # Verify dependency directive would be added
    config_with_dep = SlurmJobConfig(
        job_name="test", extra_sbatch={"dependency": "afterok:12345:67890"}
    )
    script = manager.generate_sbatch_script(config_with_dep, "echo test")
    assert "#SBATCH --dependency=afterok:12345:67890" in script


def test_submit_with_dependency_preserves_array_config():
    """Test that submit_with_dependency preserves ArrayJobConfig fields."""
    manager = SlurmJobManager(dry_run=True)
    array_config = ArrayJobConfig(
        job_name="dependent_array",
        array_size=50,
        max_concurrent=10,
        partition="compute",
    )

    # Submit with dependencies
    job_id = manager.submit_with_dependency(
        array_config,
        "python process.py $SLURM_ARRAY_TASK_ID",
        depends_on=["12345"],
        dependency_type="afterok",
    )

    assert job_id == "DRY_RUN_dependent_array"

    # Verify the generated script still contains array directive
    # We need to manually build the config with dependency to check the script
    config_with_dep = ArrayJobConfig(
        job_name="dependent_array",
        array_size=50,
        max_concurrent=10,
        partition="compute",
        extra_sbatch={"dependency": "afterok:12345"},
    )
    script = manager.generate_sbatch_script(config_with_dep, "python process.py $SLURM_ARRAY_TASK_ID")
    
    # Assert array directive is present
    assert "#SBATCH --array=0-49%10" in script
    # Assert dependency directive is present
    assert "#SBATCH --dependency=afterok:12345" in script


def test_slurm_job_state_enum():
    """Test SlurmJobState enum values."""
    assert SlurmJobState.PENDING.value == "PENDING"
    assert SlurmJobState.RUNNING.value == "RUNNING"
    assert SlurmJobState.COMPLETED.value == "COMPLETED"
    assert SlurmJobState.FAILED.value == "FAILED"
    assert SlurmJobState.CANCELLED.value == "CANCELLED"
    assert SlurmJobState.TIMEOUT.value == "TIMEOUT"
    assert SlurmJobState.UNKNOWN.value == "UNKNOWN"


def test_slurm_job_status_dataclass():
    """Test SlurmJobStatus dataclass."""
    status = SlurmJobStatus(
        job_id="12345",
        state=SlurmJobState.COMPLETED,
        submit_time="2026-02-08T10:00:00",
        start_time="2026-02-08T10:01:00",
        end_time="2026-02-08T10:15:00",
        exit_code=0,
    )

    assert status.job_id == "12345"
    assert status.state == SlurmJobState.COMPLETED
    assert status.submit_time == "2026-02-08T10:00:00"
    assert status.start_time == "2026-02-08T10:01:00"
    assert status.end_time == "2026-02-08T10:15:00"
    assert status.exit_code == 0


def test_status_dry_run():
    """Test status query in dry-run mode."""
    manager = SlurmJobManager(dry_run=True)

    # Dry run jobs always return COMPLETED
    status = manager.status("DRY_RUN_test")
    assert status.job_id == "DRY_RUN_test"
    assert status.state == SlurmJobState.COMPLETED
    assert status.exit_code == 0


def test_cancel_dry_run():
    """Test cancel in dry-run mode."""
    manager = SlurmJobManager(dry_run=True)

    result = manager.cancel("DRY_RUN_test")
    assert result is True


def test_parse_state():
    """Test state string parsing."""
    # Test various SLURM state strings
    assert SlurmJobManager._parse_state("PENDING") == SlurmJobState.PENDING
    assert SlurmJobManager._parse_state("RUNNING") == SlurmJobState.RUNNING
    assert SlurmJobManager._parse_state("COMPLETED") == SlurmJobState.COMPLETED
    assert SlurmJobManager._parse_state("FAILED") == SlurmJobState.FAILED
    assert SlurmJobManager._parse_state("CANCELLED") == SlurmJobState.CANCELLED
    assert SlurmJobManager._parse_state("TIMEOUT") == SlurmJobState.TIMEOUT
    assert SlurmJobManager._parse_state("TO") == SlurmJobState.TIMEOUT
    assert SlurmJobManager._parse_state("UNKNOWN_STATE") == SlurmJobState.UNKNOWN

    # Test case insensitivity
    assert SlurmJobManager._parse_state("pending") == SlurmJobState.PENDING
    assert SlurmJobManager._parse_state("running") == SlurmJobState.RUNNING
    
    # Test terminal states that should map to FAILED/CANCELLED
    assert SlurmJobManager._parse_state("OUT_OF_MEMORY") == SlurmJobState.FAILED
    assert SlurmJobManager._parse_state("OOM") == SlurmJobState.FAILED
    assert SlurmJobManager._parse_state("NODE_FAIL") == SlurmJobState.FAILED
    assert SlurmJobManager._parse_state("NODE_FAILURE") == SlurmJobState.FAILED
    assert SlurmJobManager._parse_state("PREEMPTED") == SlurmJobState.CANCELLED


def test_full_workflow_dry_run():
    """Test complete workflow in dry-run mode."""
    manager = SlurmJobManager(dry_run=True)

    # Submit array job
    array_config = ArrayJobConfig(
        job_name="workflow_array",
        array_size=10,
        max_concurrent=5,
        partition="compute",
        time_limit="00:30:00",
    )
    array_job_id = manager.submit(array_config, "python process_chunk.py $SLURM_ARRAY_TASK_ID")
    assert array_job_id == "DRY_RUN_workflow_array"

    # Submit dependent job
    consolidate_config = SlurmJobConfig(
        job_name="workflow_consolidate",
        partition="compute",
        time_limit="00:15:00",
    )
    consolidate_job_id = manager.submit_with_dependency(
        consolidate_config,
        "python consolidate.py",
        depends_on=[array_job_id],
    )
    assert consolidate_job_id == "DRY_RUN_workflow_consolidate"

    # Check status
    status = manager.status(consolidate_job_id)
    assert status.state == SlurmJobState.COMPLETED


def test_array_size_validation():
    """Test that array_size is validated."""
    manager = SlurmJobManager(dry_run=True)
    
    # Invalid array_size < 1
    config = ArrayJobConfig(job_name="test", array_size=0)
    try:
        manager.generate_sbatch_script(config, "echo test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "array_size must be >= 1" in str(e)
    
    # Invalid array_size < 0
    config = ArrayJobConfig(job_name="test", array_size=-5)
    try:
        manager.generate_sbatch_script(config, "echo test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "array_size must be >= 1" in str(e)


def test_max_concurrent_validation():
    """Test that max_concurrent is validated."""
    manager = SlurmJobManager(dry_run=True)
    
    # Invalid max_concurrent < 0
    config = ArrayJobConfig(job_name="test", array_size=10, max_concurrent=-1)
    try:
        manager.generate_sbatch_script(config, "echo test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "max_concurrent must be >= 0" in str(e)


def test_env_var_quoting():
    """Test that environment variable values are properly quoted."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(
        env_vars={
            "SIMPLE": "value",
            "WITH_SPACES": "value with spaces",
            "WITH_SPECIAL": "value$with'special\"chars",
        }
    )
    
    script = manager.generate_sbatch_script(config, "echo test")
    
    # Values should be quoted
    assert "export SIMPLE=value" in script
    assert "export WITH_SPACES='value with spaces'" in script
    # Special characters should be escaped or quoted
    assert "WITH_SPECIAL=" in script


def test_env_var_key_validation():
    """Test that environment variable keys are validated."""
    manager = SlurmJobManager(dry_run=True)
    
    # Invalid key with special characters
    config = SlurmJobConfig(env_vars={"INVALID;KEY": "value"})
    try:
        manager.generate_sbatch_script(config, "echo test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid environment variable name" in str(e)


def test_status_dry_run_non_dry_run_job():
    """Test status query in dry-run mode for non-DRY_RUN job IDs."""
    manager = SlurmJobManager(dry_run=True)
    
    # Non-DRY_RUN job ID should return UNKNOWN without executing commands
    status = manager.status("12345")
    assert status.job_id == "12345"
    assert status.state == SlurmJobState.UNKNOWN


def test_wait_terminates_on_unknown():
    """Test that wait() treats UNKNOWN as terminal state."""
    manager = SlurmJobManager(dry_run=True)
    
    # In dry-run mode, non-DRY_RUN job IDs return UNKNOWN
    # This should cause wait() to return immediately instead of timing out
    status = manager.wait("12345", poll_interval=0.1, timeout=1.0)
    assert status.state == SlurmJobState.UNKNOWN
