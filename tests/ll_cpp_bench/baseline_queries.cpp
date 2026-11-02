// Host-only benchmark metadata. The frozen kernel source has no experiment queries.
#include <cstdint>
extern "C" uint32_t deepep_ll_combine_small_batch_enabled()
{
    return 0U;
}
extern "C" int32_t deepep_ll_combine_recv_cores()
{
    return 0;
}
extern "C" uint32_t deepep_ll_combine_early_reduce_enabled()
{
    return 0U;
}
extern "C" uint32_t deepep_ll_combine_profile_enabled()
{
    return 0U;
}
extern "C" uint32_t deepep_ll_combine_return_options()
{
    return 0U;
}
