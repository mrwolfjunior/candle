use candle::{Device, DType, Error, Result, Tensor};

#[derive(Debug, Clone)]
pub struct InPlaceKvCache {
    k_buf: Tensor,
    v_buf: Tensor,
    current_pos: usize,
    max_seq_len: usize,
    b_sz: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl InPlaceKvCache {
    pub fn new(
        b_sz: usize,
        n_kv_head: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let k_buf = Tensor::zeros((b_sz, n_kv_head, max_seq_len, head_dim), dtype, device)?;
        let v_buf = Tensor::zeros((b_sz, n_kv_head, max_seq_len, head_dim), dtype, device)?;
        Ok(Self {
            k_buf,
            v_buf,
            current_pos: 0,
            max_seq_len,
            b_sz,
            n_kv_head,
            head_dim,
        })
    }

    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn b_sz(&self) -> usize {
        self.b_sz
    }

    pub fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn k_buf(&self) -> &Tensor {
        &self.k_buf
    }

    pub fn v_buf(&self) -> &Tensor {
        &self.v_buf
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let (kb, kh, seq_len, kd) = k.dims4()?;
        let (vb, vh, v_seq_len, vd) = v.dims4()?;

        if kb != self.b_sz || kh != self.n_kv_head || kd != self.head_dim {
            return Err(Error::Msg(format!(
                "KV cache append k shape mismatch: expected ({}, {}, seq_len, {}), got ({kb}, {kh}, {seq_len}, {kd})",
                self.b_sz, self.n_kv_head, self.head_dim
            )));
        }
        if vb != self.b_sz || vh != self.n_kv_head || vd != self.head_dim || v_seq_len != seq_len {
            return Err(Error::Msg(format!(
                "KV cache append v shape mismatch: expected ({}, {}, {seq_len}, {}), got ({vb}, {vh}, {v_seq_len}, {vd})",
                self.b_sz, self.n_kv_head, self.head_dim
            )));
        }

        if self.current_pos + seq_len > self.max_seq_len {
            return Err(Error::Msg(format!(
                "KV cache overflow: attempted to write at pos {} with len {}, but max_seq_len is {}",
                self.current_pos, seq_len, self.max_seq_len
            )));
        }

        if seq_len > 0 {
            let k_cont = if k.is_contiguous() {
                k.clone()
            } else {
                k.contiguous()?
            };
            let v_cont = if v.is_contiguous() {
                v.clone()
            } else {
                v.contiguous()?
            };
            self.k_buf.slice_set(&k_cont, 2, self.current_pos)?;
            self.v_buf.slice_set(&v_cont, 2, self.current_pos)?;
            self.current_pos += seq_len;
        }

        Ok(())
    }

    pub fn append_rolling(&mut self, k: &Tensor, v: &Tensor, window: usize) -> Result<()> {
        let (kb, kh, seq_len, kd) = k.dims4()?;
        let (vb, vh, v_seq_len, vd) = v.dims4()?;

        if kb != self.b_sz || kh != self.n_kv_head || kd != self.head_dim {
            return Err(Error::Msg(format!(
                "KV cache append k shape mismatch: expected ({}, {}, seq_len, {}), got ({kb}, {kh}, {seq_len}, {kd})",
                self.b_sz, self.n_kv_head, self.head_dim
            )));
        }
        if vb != self.b_sz || vh != self.n_kv_head || vd != self.head_dim || v_seq_len != seq_len {
            return Err(Error::Msg(format!(
                "KV cache append v shape mismatch: expected ({}, {}, {seq_len}, {}), got ({vb}, {vh}, {v_seq_len}, {vd})",
                self.b_sz, self.n_kv_head, self.head_dim
            )));
        }

        if seq_len == 0 {
            return Ok(());
        }

        let cap = window.min(self.max_seq_len);

        if seq_len >= cap {
            let k_slice = k.narrow(2, seq_len - cap, cap)?;
            let v_slice = v.narrow(2, seq_len - cap, cap)?;
            let k_cont = if k_slice.is_contiguous() {
                k_slice
            } else {
                k_slice.contiguous()?
            };
            let v_cont = if v_slice.is_contiguous() {
                v_slice
            } else {
                v_slice.contiguous()?
            };
            self.k_buf.slice_set(&k_cont, 2, 0)?;
            self.v_buf.slice_set(&v_cont, 2, 0)?;
            self.current_pos = cap;
        } else if self.current_pos + seq_len <= cap {
            let k_cont = if k.is_contiguous() {
                k.clone()
            } else {
                k.contiguous()?
            };
            let v_cont = if v.is_contiguous() {
                v.clone()
            } else {
                v.contiguous()?
            };
            self.k_buf.slice_set(&k_cont, 2, self.current_pos)?;
            self.v_buf.slice_set(&v_cont, 2, self.current_pos)?;
            self.current_pos += seq_len;
        } else {
            let evict = (self.current_pos + seq_len) - cap;
            let keep_len = self.current_pos - evict;
            let kept_k = self.k_buf.narrow(2, evict, keep_len)?.contiguous()?;
            let kept_v = self.v_buf.narrow(2, evict, keep_len)?.contiguous()?;
            self.k_buf.slice_set(&kept_k, 2, 0)?;
            self.v_buf.slice_set(&kept_v, 2, 0)?;

            let k_cont = if k.is_contiguous() {
                k.clone()
            } else {
                k.contiguous()?
            };
            let v_cont = if v.is_contiguous() {
                v.clone()
            } else {
                v.contiguous()?
            };
            self.k_buf.slice_set(&k_cont, 2, keep_len)?;
            self.v_buf.slice_set(&v_cont, 2, keep_len)?;
            self.current_pos = cap;
        }

        Ok(())
    }

    pub fn current_view(&self) -> Result<(Tensor, Tensor)> {
        if self.current_pos == 0 {
            let empty_k = Tensor::zeros(
                (self.b_sz, self.n_kv_head, 0, self.head_dim),
                self.k_buf.dtype(),
                self.k_buf.device(),
            )?;
            let empty_v = Tensor::zeros(
                (self.b_sz, self.n_kv_head, 0, self.head_dim),
                self.v_buf.dtype(),
                self.v_buf.device(),
            )?;
            Ok((empty_k, empty_v))
        } else {
            let k = self.k_buf.narrow(2, 0, self.current_pos)?;
            let v = self.v_buf.narrow(2, 0, self.current_pos)?;
            Ok((k, v))
        }
    }

    pub fn rollback(&mut self, new_pos: usize) -> Result<()> {
        if new_pos > self.current_pos {
            return Err(Error::Msg(format!(
                "Cannot rollback to position {} greater than current pos {}",
                new_pos, self.current_pos
            )));
        }
        self.current_pos = new_pos;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.current_pos = 0;
    }
}
