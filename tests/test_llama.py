from typing import Tuple
import pytest
import torch

from open_diloco.llama import GPT, CausalSelfAttention, Config, build_rope_cache
from xformers.ops.fmha.common import AttentionFwOpBase


@pytest.fixture
def config() -> Config:
    return Config(
        name="llama",
        n_embd=64,
        n_head=2,
        n_layer=2,
        vocab_size=1024,
    )


@pytest.mark.parametrize("attention_impl", ["sdpa", "xformers", "fa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gpt(config: Config, attention_impl: str, dtype: torch.dtype):
    config.attention_impl = attention_impl
    _test_gpt(config, dtype)


def _test_gpt(config: Config, dtype: torch.dtype):
    device = torch.device("cuda")
    model = GPT(config).to(dtype).to(device)

    BATCH_SIZE = 16
    SEQ_LEN = 8
    VOCAB_SIZE = 1024

    input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

    output = model(input)

    assert output is not None
    assert not output.isnan().any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attention_impl", ["xformers", "fa"])
def test_gpt_output(config: Config, dtype: torch.dtype, attention_impl: str):
    """
    in this test we compare the output of the GPT with sdpa and xformers/fa
    """

    device = torch.device("cuda")
    model = GPT(config).to(dtype).to(device)

    BATCH_SIZE = 16
    SEQ_LEN = 8
    VOCAB_SIZE = 1024

    input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

    ###  SDPA
    config.attention_impl = "sdpa"
    output_sdpa = model(input)

    ### XFORMERS
    config.attention_impl = attention_impl
    output_xformers = model(input)

    ### TESTING
    assert output_sdpa.shape == output_xformers.shape

    ### xformers has a higher tolerance
    atol = AttentionFwOpBase.ERROR_ATOL[dtype]
    rtol = AttentionFwOpBase.ERROR_RTOL[dtype]
    torch.testing.assert_close(output_sdpa, output_xformers, atol=atol, rtol=rtol)


def get_cos_and_sin_attn(config: Config, seq_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = build_rope_cache(
        seq_len=seq_len,
        n_elem=config.rope_n_elem,
        device=device,
        condense_ratio=config.rope_condense_ratio,
        base=config.rope_base,
    )
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    return cos, sin


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attention_impl", ["xformers", "fa"])
def test_attn_output(config: Config, dtype: torch.dtype, attention_impl: str):
    """
    in this test we compare the output of the GPT with sdpa and xformers/fa
    """

    device = torch.device("cuda")
    model = CausalSelfAttention(config).to(dtype).to(device)

    BATCH_SIZE = 16
    SEQ_LEN = 8

    input = torch.rand(BATCH_SIZE, SEQ_LEN, config.n_embd).to(dtype).to(device)
    cos, sin = get_cos_and_sin_attn(config, SEQ_LEN, device)

    ###  SDPA
    config.attention_impl = "sdpa"
    output_sdpa = model(input, cos, sin)

    ### XFORMERS
    config.attention_impl = attention_impl
    output_xformers = model(input, cos, sin)

    ### TESTING
    assert output_sdpa.shape == output_xformers.shape

    ### xformers has a higher tolerance
    atol = AttentionFwOpBase.ERROR_ATOL[dtype]
    rtol = AttentionFwOpBase.ERROR_RTOL[dtype]
    torch.testing.assert_close(output_sdpa, output_xformers, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attention_impl", ["xformers", "fa"])
def test_context_stuffing_attn(config: Config, dtype: torch.dtype, attention_impl: str):
    """
    In this test we compare normal pad attention with stuffing.

    input is [[2, 1, 4, 8, PAD, PAD, PAD, PAD], [1, 4, 2, 7, PAD, PAD, PAD, PAD]]
    for padded input and [[2, 1, 4, 8, 1, 4, 2, 7]] for stuffed input

    we then compare the output of the two and should be the same.
    """
    device = torch.device("cuda")
    model = CausalSelfAttention(config).to(dtype).to(device)

    config.attention_impl = attention_impl

    SEQ_LEN = 8

    emb = torch.nn.Embedding(10, config.n_embd).to(dtype).to(device)

    pad_id = 0
    input_raw = (
        torch.Tensor([[2, 1, 4, 8, pad_id, pad_id, pad_id, pad_id], [1, 4, 2, 7, pad_id, pad_id, pad_id, pad_id]])
        .long()
        .to(device)
    )
    input = emb(input_raw)

    input_stuff_raw = torch.Tensor([[2, 1, 4, 8, 1, 4, 2, 7]]).long().to(device)
    seqlens = [4, 4]
    input_stuff = emb(input_stuff_raw)

    cos, sin = get_cos_and_sin_attn(config, SEQ_LEN, device)

    ### batch
    output_ctx_stuff = model(input, cos, sin)

    output_ctx_stuff = output_ctx_stuff[:, :4, :]  # remove padding token

    output_xformers_stuff = model(input_stuff, cos, sin, seqlens=seqlens)
    output_xformers_stuff = output_xformers_stuff.reshape(2, 4, config.n_embd)

    ### TESTING
    assert output_ctx_stuff.shape == output_xformers_stuff.shape

    ### xformers has a higher tolerance
    atol = AttentionFwOpBase.ERROR_ATOL[dtype]
    rtol = AttentionFwOpBase.ERROR_RTOL[dtype]
    torch.testing.assert_close(output_ctx_stuff, output_xformers_stuff, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attention_impl", ["xformers", "fa"])
def test_context_stuffing_attn_2(config: Config, dtype: torch.dtype, attention_impl: str):
    """
    this test is slightu different from the one above, it tests
    that passing two time the same input in a stuff way yield the same results.
    """
    device = torch.device("cuda")

    config.attention_impl = attention_impl
    model = CausalSelfAttention(config).to(dtype).to(device)

    SEQ_LEN = 8

    emb = torch.nn.Embedding(10, config.n_embd).to(dtype).to(device)

    seq = [2, 1, 4, 8]
    input_stuff_raw = torch.Tensor([seq + seq]).long().to(device)
    seqlens = [len(seq), len(seq)]
    input_stuff = emb(input_stuff_raw)

    cos, sin = get_cos_and_sin_attn(config, SEQ_LEN, device)

    output = model(input_stuff, cos, sin, seqlens=seqlens)

    output_left = output[:, :4, :]
    output_right = output[:, 4:, :]

    ### TESTING
    assert output_left.shape == output_right.shape

    ### xformers has a higher tolerance
    atol = AttentionFwOpBase.ERROR_ATOL[dtype]
    rtol = AttentionFwOpBase.ERROR_RTOL[dtype]
    torch.testing.assert_close(output_left, output_right, atol=atol, rtol=rtol)
