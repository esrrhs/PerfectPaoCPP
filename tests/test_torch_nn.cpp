#include <gtest/gtest.h>
#include "algo/ppo.h"
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"

TEST(TorchNNTest, ActorForwardBackward) {
    torch_nn::ModelConfig cfg;
    cfg.hidden = 64;
    cfg.lstmHidden = 32;

    torch_nn::Actor actor(cfg);

    int64_t B = 4;
    int64_t L = 16;

    torch::Tensor xImp = torch::randn({B, torch_nn::kImpInput});
    torch::Tensor seq = torch::randn({B, torch_nn::kLstmSteps, torch_nn::kLstmIn});
    torch::Tensor actionFeat = torch::randn({L, torch_nn::kActionInput});
    torch::Tensor actionSample = torch::randint(0, B, {L}, torch::kLong);
    torch::Tensor actionId = torch::randint(0, torch_nn::kNumActions, {L}, torch::kLong);

    torch::Tensor logits = actor->forward(xImp, seq, actionFeat, actionSample, actionId);

    ASSERT_EQ(logits.size(0), B);
    ASSERT_EQ(logits.size(1), torch_nn::kNumActions);

    torch::Tensor loss = logits.sum();
    loss.backward();

    ASSERT_TRUE(actor->l1->weight.grad().defined());
}

TEST(TorchNNTest, CriticForwardBackward) {
    torch_nn::ModelConfig cfg;
    cfg.hidden = 64;
    cfg.lstmHidden = 32;

    torch_nn::Critic critic(cfg);

    int64_t B = 4;
    torch::Tensor xImp = torch::randn({B, torch_nn::kImpInput});
    torch::Tensor seq = torch::randn({B, torch_nn::kLstmSteps, torch_nn::kLstmIn});
    torch::Tensor extra = torch::randn({B, torch_nn::kExtraInput});

    torch::Tensor val = critic->forward(xImp, seq, extra);

    ASSERT_EQ(val.size(0), B);
    ASSERT_EQ(val.size(1), 1);

    torch::Tensor loss = val.sum();
    loss.backward();

    ASSERT_TRUE(critic->c1->weight.grad().defined());
}

TEST(TorchNNTest, PPOLearnerStep) {
    torch_nn::ModelConfig cfg;
    cfg.hidden = 64;
    cfg.lstmHidden = 32;

    torch_nn::Actor actor(cfg);
    torch_nn::Critic critic(cfg);

    torch_nn::PPOConfig ppoCfg;
    ppoCfg.epochs = 2;
    ppoCfg.minibatch = 4;

    torch_nn::PPOLearner learner(actor, critic, ppoCfg);

    int64_t B = 8;
    int64_t L = 24;

    torch_nn::BatchTensors bt;
    bt.xImp = torch::randn({B, torch_nn::kImpInput});
    bt.seq = torch::randn({B, torch_nn::kLstmSteps, torch_nn::kLstmIn});
    bt.extra = torch::randn({B, torch_nn::kExtraInput});
    bt.actions = torch::randint(0, torch_nn::kNumActions, {B}, torch::kLong);
    bt.oldLogp = torch::randn({B});
    bt.adv = torch::randn({B});
    bt.ret = torch::randn({B});
    bt.oldValue = torch::randn({B});

    bt.actionFeat = torch::randn({L, torch_nn::kActionInput});
    bt.actionSample = torch::randint(0, B, {L}, torch::kLong);
    bt.actionId = torch::randint(0, torch_nn::kNumActions, {L}, torch::kLong);

    auto stats = learner.update(bt);
    ASSERT_GT(stats.epochsCompleted, 0);
    ASSERT_TRUE(std::isfinite(stats.pgLoss));
    ASSERT_TRUE(std::isfinite(stats.vLoss));
}

TEST(TorchNNTest, ValueLossScaling) {
    torch_nn::ModelConfig cfg;
    cfg.hidden = 32;
    cfg.lstmHidden = 16;

    torch_nn::Actor actor(cfg);
    torch_nn::Critic critic(cfg);

    torch_nn::PPOConfig ppoCfg1;
    ppoCfg1.epochs = 1;
    ppoCfg1.minibatch = 4;
    ppoCfg1.vfCoef = 0.5f;

    torch_nn::PPOConfig ppoCfg2;
    ppoCfg2.epochs = 1;
    ppoCfg2.minibatch = 4;
    ppoCfg2.vfCoef = 1.0f;

    torch_nn::PPOLearner learner1(actor, critic, ppoCfg1);
    torch_nn::PPOLearner learner2(actor, critic, ppoCfg2);

    int64_t B = 4;
    int64_t L = 8;
    torch_nn::BatchTensors bt1;
    bt1.xImp = torch::zeros({B, torch_nn::kImpInput});
    bt1.seq = torch::zeros({B, torch_nn::kLstmSteps, torch_nn::kLstmIn});
    bt1.extra = torch::zeros({B, torch_nn::kExtraInput});
    bt1.actions = torch::zeros({B}, torch::kLong);
    bt1.oldLogp = torch::zeros({B});
    bt1.adv = torch::zeros({B});
    bt1.ret = torch::ones({B}) * 2.0f;
    bt1.oldValue = torch::zeros({B});
    bt1.actionFeat = torch::zeros({L, torch_nn::kActionInput});
    bt1.actionSample = torch::zeros({L}, torch::kLong);
    bt1.actionId = torch::zeros({L}, torch::kLong);

    auto bt2 = bt1;
    auto stats1 = learner1.update(bt1);
    auto stats2 = learner2.update(bt2);

    // vLoss with vfCoef=1.0 should be roughly 2x vLoss with vfCoef=0.5
    EXPECT_NEAR(stats2.vLoss, stats1.vLoss * 2.0, std::max(1e-6, stats1.vLoss * 0.1));
}

TEST(TorchNNTest, GAESparseGameIds) {
    std::vector<algo::Transition> tr(4);
    tr[0].gameId = 1000;
    tr[0].reward = 0.0f;
    tr[0].value = 1.0f;
    tr[0].terminal = false;

    tr[1].gameId = 1000;
    tr[1].reward = 2.0f;
    tr[1].value = 2.0f;
    tr[1].terminal = true;

    tr[2].gameId = 999999;
    tr[2].reward = -1.0f;
    tr[2].value = 0.0f;
    tr[2].terminal = true;

    tr[3].gameId = -1;  // unassigned / skipped

    algo::assignEpisodeReturns(tr, 1.0f, 0.95f);

    EXPECT_FLOAT_EQ(tr[1].ret, 2.0f);
    EXPECT_FLOAT_EQ(tr[1].adv, 0.0f);
    EXPECT_FLOAT_EQ(tr[0].ret, 2.0f);
    EXPECT_FLOAT_EQ(tr[0].adv, 1.0f);
    EXPECT_FLOAT_EQ(tr[2].ret, -1.0f);
    EXPECT_FLOAT_EQ(tr[2].adv, -1.0f);
}

TEST(TorchNNTest, ModelSaveLoadRoundtrip) {
    torch_nn::ModelConfig cfg;
    cfg.hidden = 32;
    cfg.lstmHidden = 16;

    torch_nn::Actor a1(cfg);
    torch_nn::Critic c1(cfg);
    a1->save("test_actor_rt.pt");
    c1->save("test_critic_rt.pt");

    torch_nn::Actor a2(cfg);
    torch_nn::Critic c2(cfg);
    a2->load("test_actor_rt.pt");
    c2->load("test_critic_rt.pt");
    for (size_t i = 0; i < a1->parameters().size(); ++i) {
        ASSERT_TRUE(torch::all(a1->parameters()[i] == a2->parameters()[i]).item<bool>());
    }
    for (size_t i = 0; i < c1->parameters().size(); ++i) {
        ASSERT_TRUE(torch::all(c1->parameters()[i] == c2->parameters()[i]).item<bool>());
    }
    std::remove("test_actor_rt.pt");
    std::remove("test_critic_rt.pt");
}
