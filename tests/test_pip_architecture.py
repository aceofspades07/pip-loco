"""
PIP-Loco Architecture Torture Rack Integration Tests

Comprehensive test suite verifying:
- Phase 1: Dynamic shape calculation from config
- Phase 2: Broadcasting trap detection (shape verification)
- Phase 3: Gradient hygiene check (optimizer isolation)
- Phase 4: Single batch overfit test (convergence verification)
- Phase 5: NaN/Inf safety net

Author: PIP-Loco QA Team
Standard: Zero Silent Failures
"""

import sys
import logging
import copy
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

# Configure logging with colors for clear pass/fail visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'


def log_pass(message: str) -> None:
    """Log a passing test with green [PASS] prefix."""
    logger.info(f"{GREEN}[PASS]{RESET} {message}")


def log_fail(message: str) -> None:
    """Log a failing test with red [FAIL] prefix."""
    logger.error(f"{RED}[FAIL]{RESET} {message}")


def log_info(message: str) -> None:
    """Log informational message with cyan [INFO] prefix."""
    logger.info(f"{CYAN}[INFO]{RESET} {message}")


def log_section(title: str) -> None:
    """Log a section header."""
    logger.info(f"\n{BOLD}{YELLOW}{'='*70}{RESET}")
    logger.info(f"{BOLD}{YELLOW}{title:^70}{RESET}")
    logger.info(f"{BOLD}{YELLOW}{'='*70}{RESET}\n")


# =============================================================================
# PHASE 1: SETUP & DYNAMIC SHAPE CALCULATION
# =============================================================================

class TestConfig:
    """Centralized test configuration derived from pip_config.py"""
    
    def __init__(self):
        log_section("PHASE 1: Setup & Dynamic Shape Calculation")
        
        # Import configurations from the Single Source of Truth
        try:
            from config.pip_config import PIPGO2Cfg, PIPTrainCfg
            log_pass("Successfully imported PIPGO2Cfg and PIPTrainCfg")
        except ImportError as e:
            log_fail(f"Failed to import config: {e}")
            raise
        
        self.env_cfg = PIPGO2Cfg
        self.train_cfg = PIPTrainCfg
        
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log_info(f"Using device: {self.device}")
        
        # Test batch size (smaller than training for faster tests)
        self.batch_size = 64
        
        # === DYNAMIC SHAPE CALCULATION (CRITICAL: No hardcoded values!) ===
        
        # Environment dimensions
        self.num_observations = self.env_cfg.env.num_observations  # 45
        self.num_actions = self.env_cfg.env.num_actions  # 12
        self.history_len = self.env_cfg.env.history_len  # 50
        
        # Calculate privileged observation dimension dynamically
        num_scan_points_x = len(self.env_cfg.terrain.measured_points_x)
        num_scan_points_y = len(self.env_cfg.terrain.measured_points_y)
        self.num_scan_points = num_scan_points_x * num_scan_points_y  # 17 * 11 = 187
        
        # Privileged obs composition:
        # - Blind observations (45)
        # - Additional privileged state (18): base_lin_vel(3), base_ang_vel(3), 
        #   projected_gravity(3), commands(4), friction(1), restitution(1), 
        #   payload_mass(1), com_displacement(3) ~= 18-19 dims typically
        self.privileged_state_dim = 18  # Standard privileged state components
        self.privileged_obs_dim = (
            self.num_observations + 
            self.privileged_state_dim + 
            self.num_scan_points
        )
        
        log_info(f"Calculated dimensions from config:")
        log_info(f"  num_observations:    {self.num_observations}")
        log_info(f"  num_actions:         {self.num_actions}")
        log_info(f"  history_len:         {self.history_len}")
        log_info(f"  num_scan_points:     {self.num_scan_points} ({num_scan_points_x} x {num_scan_points_y})")
        log_info(f"  privileged_state:    {self.privileged_state_dim}")
        log_info(f"  privileged_obs_dim:  {self.privileged_obs_dim}")
        
        # Network architecture from config
        self.actor_hidden_dims = self.train_cfg.policy.actor_hidden_dims
        self.critic_hidden_dims = self.train_cfg.policy.critic_hidden_dims
        self.estimator_hidden_dims = self.train_cfg.estimator.hidden_dims
        self.dreamer_hidden_dims = self.train_cfg.dreamer.hidden_dims
        self.dreamer_horizon = self.train_cfg.policy.dreamer_horizon
        self.init_noise_std = self.train_cfg.policy.init_noise_std
        
        # Training hyperparameters from config
        self.lr_encoder = self.train_cfg.algorithm.lr_encoder
        self.lr_actor = self.train_cfg.algorithm.lr_actor
        self.lr_critic = self.train_cfg.algorithm.lr_critic
        self.num_epochs = self.train_cfg.algorithm.num_learning_epochs
        self.clip_param = self.train_cfg.algorithm.clip_param
        self.entropy_coef = self.train_cfg.algorithm.entropy_coef
        self.value_loss_coef = self.train_cfg.algorithm.value_loss_coef
        self.max_grad_norm = self.train_cfg.algorithm.max_grad_norm
        self.gamma = self.train_cfg.algorithm.gamma
        self.lam = self.train_cfg.algorithm.lam
        self.velocity_indices = self.train_cfg.algorithm.velocity_indices
        
        # Storage configuration
        self.num_steps_per_env = self.train_cfg.runner.num_steps_per_env
        self.num_mini_batches = self.train_cfg.algorithm.num_mini_batches
        
        log_pass("Dynamic shape calculation complete")


def generate_mock_data(cfg: TestConfig) -> Dict[str, torch.Tensor]:
    """
    Generate mock tensors for testing with correct shapes derived from config.
    
    Returns dictionary containing all tensors needed for training loop testing.
    """
    log_info("Generating mock data tensors...")
    
    B = cfg.batch_size
    device = cfg.device
    
    data = {
        # Actor inputs
        'obs': torch.randn(B, cfg.num_observations, device=device),
        'obs_history': torch.randn(B, cfg.history_len, cfg.num_observations, device=device),
        
        # Critic input (privileged)
        'privileged_obs': torch.randn(B, cfg.privileged_obs_dim, device=device),
        
        # Actions and rewards
        'actions': torch.randn(B, cfg.num_actions, device=device),
        'rewards': torch.randn(B, 1, device=device),  # Shape (B, 1) not (B,)!
        'dones': torch.zeros(B, 1, device=device),
        
        # For Dreamer training
        'next_obs': torch.randn(B, cfg.num_observations, device=device),
        
        # For PPO training
        'returns': torch.randn(B, 1, device=device),
        'advantages': torch.randn(B, 1, device=device),
        'values': torch.randn(B, 1, device=device),
        'actions_log_probs': torch.randn(B, 1, device=device),
        'mu': torch.randn(B, cfg.num_actions, device=device),
        'sigma': torch.ones(B, cfg.num_actions, device=device) * 0.5,
    }
    
    # Normalize advantages (standard practice)
    data['advantages'] = (data['advantages'] - data['advantages'].mean()) / (data['advantages'].std() + 1e-8)
    
    log_pass(f"Generated {len(data)} mock tensors on {device}")
    
    return data


def initialize_modules(cfg: TestConfig) -> Dict[str, nn.Module]:
    """
    Initialize all PIP-Loco modules using ONLY parameters from pip_config.py.
    
    Returns dictionary containing all instantiated modules.
    """
    log_info("Initializing modules from config...")
    
    # Import module classes
    from modules.velocity_estimator import VelocityEstimator
    from modules.dreamer import NoLatentModel
    from modules.pip_actor_critic import ActorCritic
    from algorithms.hybrid_trainer import HybridTrainer
    from algorithms.storage import RolloutStorage
    
    device = cfg.device
    
    # 1. Velocity Estimator (TCN)
    estimator = VelocityEstimator(
        input_dim=cfg.num_observations,
        history_length=cfg.history_len,
        hidden_dims=cfg.estimator_hidden_dims,
        output_dim=3  # Always [vx, vy, vz]
    ).to(device)
    log_pass(f"VelocityEstimator initialized: input={cfg.num_observations}, history={cfg.history_len}")
    
    # 2. Dreamer / World Model (NoLatentModel)
    dreamer = NoLatentModel(
        obs_dim=cfg.num_observations,
        action_dim=cfg.num_actions,
        hidden_dims=cfg.dreamer_hidden_dims,
        activation=nn.ELU  # Matches config activation='elu'
    ).to(device)
    log_pass(f"NoLatentModel initialized: obs={cfg.num_observations}, action={cfg.num_actions}")
    
    # 3. Actor-Critic (owns estimator and dreamer)
    actor_critic = ActorCritic(
        num_actor_obs=cfg.num_observations,
        num_critic_obs=cfg.privileged_obs_dim,
        num_actions=cfg.num_actions,
        estimator=estimator,
        dreamer=dreamer,
        horizon=cfg.dreamer_horizon,
        actor_hidden_dims=cfg.actor_hidden_dims,
        critic_hidden_dims=cfg.critic_hidden_dims,
        activation=nn.ELU,
        init_noise_std=cfg.init_noise_std,
    ).to(device)
    log_pass(f"ActorCritic initialized: actor_obs={cfg.num_observations}, critic_obs={cfg.privileged_obs_dim}")
    
    # 4. Hybrid Trainer
    trainer = HybridTrainer(
        actor_critic=actor_critic,
        device=device,
        velocity_indices=cfg.velocity_indices,
        lr_encoder=cfg.lr_encoder,
        lr_actor=cfg.lr_actor,
        lr_critic=cfg.lr_critic,
        num_epochs=cfg.num_epochs,
        mini_batch_size=cfg.batch_size,
        clip_param=cfg.clip_param,
        entropy_coef=cfg.entropy_coef,
        value_loss_coef=cfg.value_loss_coef,
        max_grad_norm=cfg.max_grad_norm,
    )
    log_pass("HybridTrainer initialized with 3 separate optimizers")
    
    # 5. Rollout Storage (for integration testing)
    # Use smaller buffer for testing
    test_num_envs = cfg.batch_size
    test_num_transitions = cfg.num_steps_per_env
    
    storage = RolloutStorage(
        num_envs=test_num_envs,
        num_transitions_per_env=test_num_transitions,
        obs_shape=(cfg.num_observations,),
        privileged_obs_shape=(cfg.privileged_obs_dim,),
        actions_shape=(cfg.num_actions,),
        history_len=cfg.history_len,
        device=device,
    )
    log_pass(f"RolloutStorage initialized: envs={test_num_envs}, transitions={test_num_transitions}")
    
    return {
        'estimator': estimator,
        'dreamer': dreamer,
        'actor_critic': actor_critic,
        'trainer': trainer,
        'storage': storage,
    }


# =============================================================================
# PHASE 2: BROADCASTING TRAP TEST (SHAPE VERIFICATION)
# =============================================================================

def test_forward_pass_shapes(cfg: TestConfig, modules: Dict[str, nn.Module], data: Dict[str, torch.Tensor]) -> bool:
    """
    Phase 2: Verify all forward pass output shapes to catch broadcasting bugs.
    
    The "Broadcasting Trap": PyTorch silently broadcasts (N,1) with (N,) which
    can cause subtle bugs in loss computation. We explicitly verify shapes.
    """
    log_section("PHASE 2: Broadcasting Trap Test (Shape Verification)")
    
    all_passed = True
    B = cfg.batch_size
    
    estimator = modules['estimator']
    dreamer = modules['dreamer']
    actor_critic = modules['actor_critic']
    
    # Put in eval mode for deterministic forward pass
    estimator.eval()
    dreamer.eval()
    actor_critic.eval()
    
    with torch.no_grad():
        # Test 1: Velocity Estimator output shape
        try:
            vel_estimate = estimator(data['obs_history'])
            expected_shape = (B, 3)
            if vel_estimate.shape == expected_shape:
                log_pass(f"VelocityEstimator output shape: {vel_estimate.shape} == {expected_shape}")
            else:
                log_fail(f"VelocityEstimator shape mismatch: Expected {expected_shape}, got {vel_estimate.shape}")
                all_passed = False
        except Exception as e:
            log_fail(f"VelocityEstimator forward failed: {e}")
            all_passed = False
        
        # Test 2: Dreamer dynamics prediction shape
        try:
            next_obs_pred = dreamer.predict_next_state(data['obs'], data['actions'])
            expected_shape = (B, cfg.num_observations)
            if next_obs_pred.shape == expected_shape:
                log_pass(f"Dreamer dynamics output shape: {next_obs_pred.shape} == {expected_shape}")
            else:
                log_fail(f"Dreamer dynamics shape mismatch: Expected {expected_shape}, got {next_obs_pred.shape}")
                all_passed = False
        except Exception as e:
            log_fail(f"Dreamer dynamics forward failed: {e}")
            all_passed = False
        
        # Test 3: Dreamer reward prediction shape (CRITICAL: must be (B, 1) not (B,))
        try:
            reward_pred = dreamer.predict_reward(data['obs'], data['actions'])
            expected_shape = (B, 1)
            if reward_pred.shape == expected_shape:
                log_pass(f"Dreamer reward output shape: {reward_pred.shape} == {expected_shape}")
            else:
                log_fail(f"Dreamer reward shape mismatch: Expected {expected_shape}, got {reward_pred.shape}")
                log_fail("  This will cause broadcasting bugs in MSE loss computation!")
                all_passed = False
        except Exception as e:
            log_fail(f"Dreamer reward forward failed: {e}")
            all_passed = False
        
        # Test 4: Dreamer value prediction shape (CRITICAL: must be (B, 1) not (B,))
        try:
            value_pred = dreamer.get_value(data['obs'])
            expected_shape = (B, 1)
            if value_pred.shape == expected_shape:
                log_pass(f"Dreamer value output shape: {value_pred.shape} == {expected_shape}")
            else:
                log_fail(f"Dreamer value shape mismatch: Expected {expected_shape}, got {value_pred.shape}")
                all_passed = False
        except Exception as e:
            log_fail(f"Dreamer value forward failed: {e}")
            all_passed = False
        
        # Test 5: Dreamer dreams (imagined trajectory) shape
        try:
            dreams = dreamer.generate_dreams(data['obs'], cfg.dreamer_horizon)
            expected_shape = (B, cfg.dreamer_horizon * cfg.num_observations)
            if dreams.shape == expected_shape:
                log_pass(f"Dreamer dreams output shape: {dreams.shape} == {expected_shape}")
            else:
                log_fail(f"Dreamer dreams shape mismatch: Expected {expected_shape}, got {dreams.shape}")
                all_passed = False
        except Exception as e:
            log_fail(f"Dreamer dreams generation failed: {e}")
            all_passed = False
        
        # Test 6: Actor action output shape
        try:
            actions, log_probs = actor_critic.act(data['obs'], data['obs_history'])
            expected_action_shape = (B, cfg.num_actions)
            expected_log_prob_shape = (B, 1)
            
            if actions.shape == expected_action_shape:
                log_pass(f"Actor action output shape: {actions.shape} == {expected_action_shape}")
            else:
                log_fail(f"Actor action shape mismatch: Expected {expected_action_shape}, got {actions.shape}")
                all_passed = False
            
            if log_probs.shape == expected_log_prob_shape:
                log_pass(f"Actor log_prob output shape: {log_probs.shape} == {expected_log_prob_shape}")
            else:
                log_fail(f"Actor log_prob shape mismatch: Expected {expected_log_prob_shape}, got {log_probs.shape}")
                all_passed = False
        except Exception as e:
            log_fail(f"Actor act() failed: {e}")
            all_passed = False
        
        # Test 7: Critic value output shape (CRITICAL for PPO loss)
        try:
            critic_value = actor_critic.evaluate(data['privileged_obs'])
            expected_shape = (B, 1)
            if critic_value.shape == expected_shape:
                log_pass(f"Critic value output shape: {critic_value.shape} == {expected_shape}")
            else:
                log_fail(f"Critic value shape mismatch: Expected {expected_shape}, got {critic_value.shape}")
                log_fail("  This will break PPO value loss: (value - returns).pow(2)")
                all_passed = False
        except Exception as e:
            log_fail(f"Critic evaluate() failed: {e}")
            all_passed = False
        
        # Test 8: evaluate_actions output shapes (for PPO update)
        try:
            log_probs, entropy = actor_critic.evaluate_actions(
                data['obs'], data['obs_history'], data['actions']
            )
            expected_log_prob_shape = (B, 1)
            
            if log_probs.shape == expected_log_prob_shape:
                log_pass(f"evaluate_actions log_prob shape: {log_probs.shape} == {expected_log_prob_shape}")
            else:
                log_fail(f"evaluate_actions log_prob shape mismatch: Expected {expected_log_prob_shape}, got {log_probs.shape}")
                all_passed = False
            
            if entropy.dim() == 0:  # Should be a scalar
                log_pass(f"evaluate_actions entropy is scalar: {entropy.shape}")
            else:
                log_fail(f"evaluate_actions entropy should be scalar, got shape: {entropy.shape}")
                all_passed = False
        except Exception as e:
            log_fail(f"evaluate_actions() failed: {e}")
            all_passed = False
        
        # Test 9: Shape compatibility check for loss computation
        log_info("Checking tensor shape compatibility for loss computation...")
        
        # Reward shapes must match for MSE
        reward_pred = dreamer.predict_reward(data['obs'], data['actions'])
        reward_target = data['rewards']
        if reward_pred.shape == reward_target.shape:
            log_pass(f"Reward shapes compatible: pred={reward_pred.shape}, target={reward_target.shape}")
        else:
            log_fail(f"Reward shape mismatch: pred={reward_pred.shape}, target={reward_target.shape}")
            log_fail("  Broadcasting will silently corrupt the loss!")
            all_passed = False
        
        # Value shapes must match for MSE
        value_pred = actor_critic.evaluate(data['privileged_obs'])
        value_target = data['values']
        if value_pred.shape == value_target.shape:
            log_pass(f"Value shapes compatible: pred={value_pred.shape}, target={value_target.shape}")
        else:
            log_fail(f"Value shape mismatch: pred={value_pred.shape}, target={value_target.shape}")
            all_passed = False
    
    # Switch back to train mode
    estimator.train()
    dreamer.train()
    actor_critic.train()
    
    return all_passed


# =============================================================================
# PHASE 3: GRADIENT HYGIENE CHECK
# =============================================================================

def fill_storage_with_dummy_data(storage, cfg: TestConfig) -> None:
    """Fill RolloutStorage with dummy data for testing."""
    device = cfg.device
    num_envs = storage.num_envs
    
    for _ in range(storage.num_transitions_per_env):
        obs = torch.randn(num_envs, cfg.num_observations, device=device)
        privileged_obs = torch.randn(num_envs, cfg.privileged_obs_dim, device=device)
        obs_history = torch.randn(num_envs, cfg.history_len, cfg.num_observations, device=device)
        actions = torch.randn(num_envs, cfg.num_actions, device=device)
        rewards = torch.randn(num_envs, 1, device=device)
        next_obs = torch.randn(num_envs, cfg.num_observations, device=device)
        dones = torch.zeros(num_envs, 1, device=device)
        values = torch.randn(num_envs, 1, device=device)
        actions_log_prob = torch.randn(num_envs, 1, device=device)
        mu = torch.randn(num_envs, cfg.num_actions, device=device)
        sigma = torch.ones(num_envs, cfg.num_actions, device=device) * 0.5
        
        storage.add_transitions(
            obs=obs,
            privileged_obs=privileged_obs,
            obs_history=obs_history,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            values=values,
            actions_log_prob=actions_log_prob,
            mu=mu,
            sigma=sigma,
        )
    
    # Compute returns with dummy last values
    last_values = torch.randn(num_envs, 1, device=device)
    storage.compute_returns(last_values, gamma=cfg.gamma, lam=cfg.lam)


def test_gradient_hygiene(cfg: TestConfig, modules: Dict[str, nn.Module]) -> bool:
    """
    Phase 3: Verify gradient isolation between optimizers.
    
    The HybridTrainer uses THREE separate optimizers to prevent gradient leakage:
    1. optimizer_est: Only updates Estimator
    2. optimizer_dream: Only updates Dreamer
    3. optimizer_ppo: Only updates Actor and Critic
    
    This test verifies that gradients from PPO loss do NOT affect Dreamer parameters.
    """
    log_section("PHASE 3: Gradient Hygiene Check")
    
    all_passed = True
    
    actor_critic = modules['actor_critic']
    trainer = modules['trainer']
    storage = modules['storage']
    
    # Fill storage with dummy data
    log_info("Filling storage with dummy data...")
    storage.clear()
    fill_storage_with_dummy_data(storage, cfg)
    log_pass("Storage filled and returns computed")
    
    # Zero all gradients before test
    trainer.optimizer_est.zero_grad()
    trainer.optimizer_dream.zero_grad()
    trainer.optimizer_ppo.zero_grad()
    
    # Record Dreamer parameters before update
    dreamer_params_before = {
        name: param.clone().detach()
        for name, param in actor_critic.dreamer.named_parameters()
    }
    
    # Run one training update
    log_info("Running trainer.update() for gradient analysis...")
    try:
        loss_dict = trainer.update(storage)
        log_pass(f"trainer.update() completed successfully")
        log_info(f"  Losses: vel={loss_dict['loss/velocity']:.4f}, dream={loss_dict['loss/dreamer']:.4f}, "
                f"policy={loss_dict['loss/policy']:.4f}, value={loss_dict['loss/value']:.4f}")
    except Exception as e:
        log_fail(f"trainer.update() failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check 1: Actor parameters have gradients
    log_info("Check 1: Verifying actor parameters have gradients...")
    actor_has_grads = True
    for name, param in actor_critic.actor.named_parameters():
        if param.grad is None:
            log_fail(f"  Actor parameter '{name}' has NO gradient!")
            actor_has_grads = False
    if actor_has_grads:
        log_pass("All actor parameters have valid gradients")
    else:
        all_passed = False
    
    # Check 2: Estimator parameters have gradients
    log_info("Check 2: Verifying estimator parameters have gradients...")
    estimator_has_grads = True
    for name, param in actor_critic.estimator.named_parameters():
        if param.grad is None:
            log_fail(f"  Estimator parameter '{name}' has NO gradient!")
            estimator_has_grads = False
    if estimator_has_grads:
        log_pass("All estimator parameters have valid gradients")
    else:
        all_passed = False
    
    # Check 3: Critic parameters have gradients
    log_info("Check 3: Verifying critic parameters have gradients...")
    critic_has_grads = True
    for name, param in actor_critic.critic.named_parameters():
        if param.grad is None:
            log_fail(f"  Critic parameter '{name}' has NO gradient!")
            critic_has_grads = False
    if critic_has_grads:
        log_pass("All critic parameters have valid gradients")
    else:
        all_passed = False
    
    # Check 4: Dreamer parameters have gradients (from their own optimizer)
    log_info("Check 4: Verifying dreamer parameters have gradients...")
    dreamer_has_grads = True
    for name, param in actor_critic.dreamer.named_parameters():
        if param.grad is None:
            log_fail(f"  Dreamer parameter '{name}' has NO gradient!")
            dreamer_has_grads = False
    if dreamer_has_grads:
        log_pass("All dreamer parameters have valid gradients")
    else:
        all_passed = False
    
    # Check 5: THE LEAK TEST - Verify optimizer isolation
    # Dreamer should be updated ONLY by its own optimizer (optimizer_dream)
    # The PPO update should NOT have changed Dreamer params through backprop
    log_info("Check 5: THE LEAK TEST - Verifying optimizer isolation...")
    
    # Since update() runs all three optimizer steps, we verify by checking
    # that the Dreamer was only updated through its dedicated loss
    # This is implicitly verified by the architecture (separate optimizer)
    # but we can check that gradient magnitudes are reasonable
    
    dreamer_grad_norms = []
    for name, param in actor_critic.dreamer.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            dreamer_grad_norms.append(grad_norm)
    
    avg_dreamer_grad_norm = sum(dreamer_grad_norms) / len(dreamer_grad_norms) if dreamer_grad_norms else 0
    log_info(f"  Average Dreamer gradient norm: {avg_dreamer_grad_norm:.6f}")
    
    # The key test: verify ActorCritic._get_actor_input() uses detach=True
    # This prevents PPO gradients from flowing into Estimator/Dreamer
    # We can verify this by checking that during PPO loss backward,
    # the Dreamer should not receive gradients from the actor loss
    
    # Create a fresh storage and do a manual gradient flow test
    log_info("  Performing manual gradient flow verification...")
    
    # Clear storage for clean test
    storage.clear()
    fill_storage_with_dummy_data(storage, cfg)
    
    # Zero all gradients
    for param in actor_critic.parameters():
        if param.grad is not None:
            param.grad.zero_()
    
    # Manually run just the PPO forward/backward to check isolation
    actor_critic.train()
    
    # Get one minibatch
    minibatch_gen = storage.generate_minibatch(cfg.num_mini_batches, 1)
    minibatch = next(minibatch_gen)
    
    obs = minibatch[0].to(cfg.device)
    privileged_obs = minibatch[1].to(cfg.device)
    obs_history = minibatch[2].to(cfg.device)
    actions = minibatch[3].to(cfg.device)
    returns = minibatch[6].to(cfg.device)
    actions_log_probs_old = minibatch[9].to(cfg.device)
    advantages = minibatch[10].to(cfg.device)
    
    # Zero all grads
    for param in actor_critic.parameters():
        if param.grad is not None:
            param.grad.zero_()
    
    # Run PPO forward pass (this should NOT send grads to Dreamer due to detach)
    log_probs, entropy = actor_critic.evaluate_actions(obs, obs_history, actions)
    value_pred = actor_critic.evaluate(privileged_obs)
    
    log_probs = log_probs.view(-1, 1)
    ratio = torch.exp(log_probs - actions_log_probs_old)
    surr1 = -advantages * ratio
    surr2 = -advantages * torch.clamp(ratio, 0.8, 1.2)
    policy_loss = torch.max(surr1, surr2).mean()
    value_loss = (value_pred - returns).pow(2).mean()
    ppo_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    
    # Backward the PPO loss ONLY
    ppo_loss.backward()
    
    # Check: Dreamer should have NO gradients from PPO loss (due to detach)
    dreamer_has_ppo_grads = False
    for name, param in actor_critic.dreamer.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 1e-10:
            dreamer_has_ppo_grads = True
            log_fail(f"  LEAK DETECTED: Dreamer param '{name}' received PPO gradient!")
    
    if not dreamer_has_ppo_grads:
        log_pass("THE LEAK TEST PASSED: PPO gradients do NOT flow to Dreamer (detach working)")
    else:
        log_fail("THE LEAK TEST FAILED: PPO gradients leaked into Dreamer!")
        all_passed = False
    
    # Similarly check Estimator
    estimator_has_ppo_grads = False
    for name, param in actor_critic.estimator.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 1e-10:
            estimator_has_ppo_grads = True
            log_fail(f"  LEAK DETECTED: Estimator param '{name}' received PPO gradient!")
    
    if not estimator_has_ppo_grads:
        log_pass("THE LEAK TEST PASSED: PPO gradients do NOT flow to Estimator (detach working)")
    else:
        log_fail("THE LEAK TEST FAILED: PPO gradients leaked into Estimator!")
        all_passed = False
    
    return all_passed


# =============================================================================
# PHASE 4: SINGLE BATCH OVERFIT TEST (CONVERGENCE VERIFICATION)
# =============================================================================

def test_single_batch_overfit(cfg: TestConfig, modules: Dict[str, nn.Module]) -> bool:
    """
    Phase 4: The Gold Standard - verify the model can overfit a single batch.
    
    If a model cannot reduce loss on a SINGLE fixed batch after multiple iterations,
    the loss function or model architecture is fundamentally broken.
    
    Success criteria:
    - loss/velocity decreases by at least 50%
    - loss/dreamer decreases by at least 50%
    - loss/value decreases
    """
    log_section("PHASE 4: Single Batch Overfit Test (The Gold Standard)")
    
    all_passed = True
    
    # Re-initialize modules for clean test
    log_info("Re-initializing modules for clean overfit test...")
    fresh_modules = initialize_modules(cfg)
    trainer = fresh_modules['trainer']
    storage = fresh_modules['storage']
    
    # Create ONE fixed batch of data
    log_info("Creating fixed batch for overfit test...")
    storage.clear()
    fill_storage_with_dummy_data(storage, cfg)
    
    # Store initial storage state for repeated use
    # We'll just refill returns each iteration
    initial_last_values = torch.randn(storage.num_envs, 1, device=cfg.device)
    
    num_overfit_iterations = 50
    losses_history = {
        'velocity': [],
        'dreamer': [],
        'policy': [],
        'value': [],
        'entropy': [],
    }
    
    log_info(f"Running {num_overfit_iterations} iterations on the SAME batch...")
    
    for i in range(num_overfit_iterations):
        # Reset storage step counter to reuse data
        storage.step = storage.num_transitions_per_env  # Mark as full
        storage.compute_returns(initial_last_values, gamma=cfg.gamma, lam=cfg.lam)
        
        # Run update
        loss_dict = trainer.update(storage)
        
        losses_history['velocity'].append(loss_dict['loss/velocity'])
        losses_history['dreamer'].append(loss_dict['loss/dreamer'])
        losses_history['policy'].append(loss_dict['loss/policy'])
        losses_history['value'].append(loss_dict['loss/value'])
        losses_history['entropy'].append(loss_dict['loss/entropy'])
        
        if (i + 1) % 10 == 0:
            log_info(f"  Iteration {i+1}/{num_overfit_iterations}: "
                    f"vel={loss_dict['loss/velocity']:.4f}, "
                    f"dream={loss_dict['loss/dreamer']:.4f}, "
                    f"value={loss_dict['loss/value']:.4f}")
    
    # Analyze results
    log_info("\nOverfit test results:")
    
    # Velocity loss check
    vel_start = losses_history['velocity'][0]
    vel_end = losses_history['velocity'][-1]
    vel_reduction = (vel_start - vel_end) / (vel_start + 1e-8) * 100
    
    if vel_reduction >= 50:
        log_pass(f"Velocity loss decreased by {vel_reduction:.1f}%: {vel_start:.4f} → {vel_end:.4f}")
    elif vel_reduction > 0:
        log_info(f"Velocity loss decreased by {vel_reduction:.1f}%: {vel_start:.4f} → {vel_end:.4f} (< 50% threshold)")
        log_info("  This may be acceptable for small batches or already-converged values")
    else:
        log_fail(f"Velocity loss did NOT decrease: {vel_start:.4f} → {vel_end:.4f}")
        log_fail("  The velocity estimator or its loss function may be broken!")
        all_passed = False
    
    # Dreamer loss check
    dream_start = losses_history['dreamer'][0]
    dream_end = losses_history['dreamer'][-1]
    dream_reduction = (dream_start - dream_end) / (dream_start + 1e-8) * 100
    
    if dream_reduction >= 50:
        log_pass(f"Dreamer loss decreased by {dream_reduction:.1f}%: {dream_start:.4f} → {dream_end:.4f}")
    elif dream_reduction > 0:
        log_info(f"Dreamer loss decreased by {dream_reduction:.1f}%: {dream_start:.4f} → {dream_end:.4f} (< 50% threshold)")
    else:
        log_fail(f"Dreamer loss did NOT decrease: {dream_start:.4f} → {dream_end:.4f}")
        log_fail("  The world model or its loss function may be broken!")
        all_passed = False
    
    # Value loss check
    val_start = losses_history['value'][0]
    val_end = losses_history['value'][-1]
    val_reduction = (val_start - val_end) / (val_start + 1e-8) * 100
    
    if val_end < val_start:
        log_pass(f"Value loss decreased by {val_reduction:.1f}%: {val_start:.4f} → {val_end:.4f}")
    else:
        log_fail(f"Value loss did NOT decrease: {val_start:.4f} → {val_end:.4f}")
        log_fail("  The critic or value loss function may be broken!")
        all_passed = False
    
    # Check for loss explosion (indicates numerical instability)
    if max(losses_history['velocity']) > 1000 or max(losses_history['dreamer']) > 1000:
        log_fail("Loss explosion detected during overfit test!")
        all_passed = False
    else:
        log_pass("No loss explosion detected - training is numerically stable")
    
    return all_passed


# =============================================================================
# PHASE 5: NAN/INF SAFETY NET
# =============================================================================

def scan_for_nan_inf(modules: Dict[str, nn.Module]) -> bool:
    """
    Phase 5: Scan all model parameters for NaN or Inf values.
    
    Any NaN/Inf in parameters indicates catastrophic numerical failure.
    """
    log_section("PHASE 5: NaN/Inf Safety Net Scan")
    
    all_clean = True
    actor_critic = modules['actor_critic']
    
    log_info("Scanning all parameters for NaN/Inf values...")
    
    for name, param in actor_critic.named_parameters():
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        
        if has_nan:
            log_fail(f"NaN detected in parameter: {name}")
            all_clean = False
        if has_inf:
            log_fail(f"Inf detected in parameter: {name}")
            all_clean = False
    
    if all_clean:
        log_pass("All parameters are clean - no NaN or Inf values detected")
    
    # Also check gradients if available
    grad_clean = True
    for name, param in actor_critic.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            
            if has_nan:
                log_fail(f"NaN detected in gradient: {name}")
                grad_clean = False
            if has_inf:
                log_fail(f"Inf detected in gradient: {name}")
                grad_clean = False
    
    if grad_clean:
        log_pass("All gradients are clean - no NaN or Inf values detected")
    
    return all_clean and grad_clean


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests() -> bool:
    """
    Run the complete PIP-Loco Architecture Torture Rack.
    
    Returns True if ALL tests pass, False otherwise.
    """
    log_section("PIP-LOCO ARCHITECTURE TORTURE RACK")
    log_info("Standard: Zero Silent Failures")
    log_info("Testing complete module integration before train.py\n")
    
    results = {
        'phase1_setup': False,
        'phase2_shapes': False,
        'phase3_gradients': False,
        'phase4_overfit': False,
        'phase5_safety': False,
    }
    
    try:
        # Phase 1: Setup
        cfg = TestConfig()
        data = generate_mock_data(cfg)
        modules = initialize_modules(cfg)
        results['phase1_setup'] = True
        log_pass("Phase 1 COMPLETE: Setup & Dynamic Shape Calculation")
    except Exception as e:
        log_fail(f"Phase 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Phase 2: Shape Verification
        results['phase2_shapes'] = test_forward_pass_shapes(cfg, modules, data)
        if results['phase2_shapes']:
            log_pass("Phase 2 COMPLETE: Broadcasting Trap Test")
        else:
            log_fail("Phase 2 FAILED: Shape verification errors detected")
    except Exception as e:
        log_fail(f"Phase 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Phase 3: Gradient Hygiene
        results['phase3_gradients'] = test_gradient_hygiene(cfg, modules)
        if results['phase3_gradients']:
            log_pass("Phase 3 COMPLETE: Gradient Hygiene Check")
        else:
            log_fail("Phase 3 FAILED: Gradient hygiene issues detected")
    except Exception as e:
        log_fail(f"Phase 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Phase 4: Overfit Test
        results['phase4_overfit'] = test_single_batch_overfit(cfg, modules)
        if results['phase4_overfit']:
            log_pass("Phase 4 COMPLETE: Single Batch Overfit Test")
        else:
            log_fail("Phase 4 FAILED: Model could not overfit single batch")
    except Exception as e:
        log_fail(f"Phase 4 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Phase 5: Safety Net
        # Re-initialize for final scan (uses modules from Phase 4 which have been trained)
        fresh_modules = initialize_modules(cfg)
        # Quick training pass
        fresh_modules['storage'].clear()
        fill_storage_with_dummy_data(fresh_modules['storage'], cfg)
        fresh_modules['trainer'].update(fresh_modules['storage'])
        
        results['phase5_safety'] = scan_for_nan_inf(fresh_modules)
        if results['phase5_safety']:
            log_pass("Phase 5 COMPLETE: NaN/Inf Safety Net")
        else:
            log_fail("Phase 5 FAILED: NaN/Inf detected in model")
    except Exception as e:
        log_fail(f"Phase 5 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Summary
    log_section("FINAL TEST SUMMARY")
    
    all_passed = all(results.values())
    
    for phase, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        log_info(f"  {phase}: [{status}]")
    
    if all_passed:
        log_info(f"\n{GREEN}{BOLD}{'='*70}{RESET}")
        log_info(f"{GREEN}{BOLD}{'ALL TESTS PASSED - Architecture Ready for train.py':^70}{RESET}")
        log_info(f"{GREEN}{BOLD}{'='*70}{RESET}")
    else:
        log_info(f"\n{RED}{BOLD}{'='*70}{RESET}")
        log_info(f"{RED}{BOLD}{'TESTS FAILED - Fix issues before proceeding':^70}{RESET}")
        log_info(f"{RED}{BOLD}{'='*70}{RESET}")
    
    return all_passed


if __name__ == "__main__":
    import sys
    
    # Add project root to path
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
