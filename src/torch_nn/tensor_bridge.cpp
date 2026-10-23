#include "torch_nn/tensor_bridge.h"
#include "algo/ppo.h"
#include <cmath>

namespace torch_nn {

BatchTensors transitionsToBatch(const std::vector<algo::Transition>& trInput,
                                float gamma,
                                float lambda,
                                torch::Device device) {
    int64_t B = static_cast<int64_t>(trInput.size());
    BatchTensors bt;
    if (B == 0) return bt;

    // Make local copy to calculate GAE / normalized advantages
    std::vector<algo::Transition> tr = trInput;
    algo::assignEpisodeReturns(tr, gamma, lambda);

    // Advantage normalization
    double meanAdv = 0.0, varAdv = 0.0;
    for (const auto& t : tr) meanAdv += t.adv;
    meanAdv /= B;
    for (const auto& t : tr) varAdv += (t.adv - float(meanAdv)) * (t.adv - float(meanAdv));
    float stdAdv = float(std::sqrt(varAdv / B + 1e-8));
    for (auto& t : tr) t.adv = (t.adv - float(meanAdv)) / stdAdv;

    // Count total action candidates across all transitions
    int64_t L = 0;
    for (const auto& t : tr) {
        L += static_cast<int64_t>(t.actions.size());
    }

    bt.xImp = torch::empty({B, kImpInput}, torch::kFloat32);
    bt.seq = torch::empty({B, kLstmSteps, kLstmIn}, torch::kFloat32);
    bt.extra = torch::empty({B, kExtraInput}, torch::kFloat32);

    bt.actions = torch::empty({B}, torch::kLong);
    bt.oldLogp = torch::empty({B}, torch::kFloat32);
    bt.adv = torch::empty({B}, torch::kFloat32);
    bt.ret = torch::empty({B}, torch::kFloat32);
    bt.oldValue = torch::empty({B}, torch::kFloat32);

    bt.actionFeat = torch::empty({L, kActionInput}, torch::kFloat32);
    bt.actionSample = torch::empty({L}, torch::kLong);
    bt.actionId = torch::empty({L}, torch::kLong);

    float* pXImp = bt.xImp.data_ptr<float>();
    float* pSeq = bt.seq.data_ptr<float>();
    float* pExtra = bt.extra.data_ptr<float>();

    int64_t* pActions = bt.actions.data_ptr<int64_t>();
    float* pOldLogp = bt.oldLogp.data_ptr<float>();
    float* pAdv = bt.adv.data_ptr<float>();
    float* pRet = bt.ret.data_ptr<float>();
    float* pOldValue = bt.oldValue.data_ptr<float>();

    float* pActFeat = bt.actionFeat.data_ptr<float>();
    int64_t* pActSample = bt.actionSample.data_ptr<int64_t>();
    int64_t* pActId = bt.actionId.data_ptr<int64_t>();

    int64_t actOffset = 0;

    for (int64_t i = 0; i < B; ++i) {
        const auto& t = tr[i];

        // 1. xImp: [kNodeBin] (uint8 -> float) + [kImpScalars]
        float* curXImp = pXImp + i * kImpInput;
        for (int j = 0; j < pdk::kNodeBin; ++j) {
            curXImp[j] = float(t.imp[j]);
        }
        std::copy(t.scalar.begin(), t.scalar.end(), curXImp + pdk::kNodeBin);

        // 2. seq: history steps of 3 consecutive moves each
        float* curSeq = pSeq + i * kLstmSteps * kLstmIn;
        for (int k = 0; k < pdk::kHistoryLen; ++k) {
            const uint8_t* src = t.imp.data() + (pdk::kStaticMatrices + k) * pdk::kCardMat;
            float* dst = curSeq + k * pdk::kCardMat;
            for (int j = 0; j < pdk::kCardMat; ++j) {
                dst[j] = float(src[j]);
            }
        }

        // 3. extra: [kExtraBin] + [kExtraScalars]
        float* curExtra = pExtra + i * kExtraInput;
        for (int j = 0; j < pdk::kExtraBin; ++j) {
            curExtra[j] = float(t.extra[j]);
        }
        std::copy(t.extraScalar.begin(), t.extraScalar.end(), curExtra + pdk::kExtraBin);

        pActions[i] = t.action;
        pOldLogp[i] = t.logp;
        pAdv[i] = t.adv;
        pRet[i] = t.ret;
        pOldValue[i] = t.value;

        for (const auto& [aid, afeat] : t.actions) {
            std::memcpy(pActFeat + actOffset * kActionInput, afeat.data(), kActionInput * sizeof(float));
            pActSample[actOffset] = i;
            pActId[actOffset] = aid;
            ++actOffset;
        }
    }

    if (device != torch::kCPU) {
        bt.xImp = bt.xImp.to(device, /*non_blocking=*/true);
        bt.seq = bt.seq.to(device, /*non_blocking=*/true);
        bt.extra = bt.extra.to(device, /*non_blocking=*/true);
        bt.actions = bt.actions.to(device, /*non_blocking=*/true);
        bt.oldLogp = bt.oldLogp.to(device, /*non_blocking=*/true);
        bt.adv = bt.adv.to(device, /*non_blocking=*/true);
        bt.ret = bt.ret.to(device, /*non_blocking=*/true);
        bt.oldValue = bt.oldValue.to(device, /*non_blocking=*/true);
        bt.actionFeat = bt.actionFeat.to(device, /*non_blocking=*/true);
        bt.actionSample = bt.actionSample.to(device, /*non_blocking=*/true);
        bt.actionId = bt.actionId.to(device, /*non_blocking=*/true);
    }

    return bt;
}

}  // namespace torch_nn
