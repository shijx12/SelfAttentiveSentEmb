from tensorboard import summary
from torch.autograd import Variable
import torch

def add_scalar_summary(summary_writer, name, value, step):
    value = unwrap_scalar_variable(value)
    summ = summary.scalar(name=name, scalar=value)
    summary_writer.add_summary(summary=summ, global_step=step)

def add_histo_summary(summary_writer, name, value, step):
    value = value.view(-1).data.cpu().numpy()
    summ = summary.histogram(name=name, values=value)
    summary_writer.add_summary(summary=summ, global_step=step)


def wrap_with_variable(tensor, volatile, cuda):
    if cuda:
        return Variable(tensor.cuda(), volatile=volatile)
    else:
        return Variable(tensor, volatile=volatile)


def unwrap_scalar_variable(var):
    if isinstance(var, Variable):
        return var.data[0]
    else:
        return var

def unwrap_variable(var):
    if isinstance(var, Variable):
        return var.data
    else:
        return var


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1), 1) + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')

def sequence_mask(sequence_length, max_length=None):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand
