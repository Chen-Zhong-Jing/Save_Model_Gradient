import numpy as np
import matplotlib.pyplot as plt
import shelve
from scipy import stats
from operator import itemgetter

def Wasserstein_distance(p, u, v):
    assert len(u) == len(v)
    return np.mean(np.abs(np.sort(u)[1:u.size - 1] - np.sort(v)[1:v.size - 1]) ** p) ** (1 / p)
def optimize_bin_edges(x, bins='fd'):
    return np.histogram_bin_edges(x, bins=bins)



# -----------------------------------------------------------------------------------------------
"Huffman encoding"


class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        # probability of symbol
        self.prob = prob

        # symbol
        self.symbol = symbol

        # left node
        self.left = left

        # right node
        self.right = right

        # tree direction (0/1)
        self.code = ''


""" A helper function to print the codes of symbols by traveling Huffman Tree"""
def Calculate_Codes(node, val=''):
    # huffman code for current node
    newVal = val + str(node.code)

    if (node.left):
        Calculate_Codes(node.left, newVal)
    if (node.right):
        Calculate_Codes(node.right, newVal)

    if (not node.left and not node.right):
        codes[node.symbol] = newVal

    return codes


""" A helper function to calculate the probabilities of symbols in given data"""


def Calculate_Probability(data):
    symbols = dict()
    for idx in range(data.size):
        symbols[str(idx)] = data[idx]
    return symbols


""" A helper function to obtain the encoded output"""


def Output_Encoded(data, coding):
    encoding_output = []
    for idx in range(data.size):
        #  print(coding[c], end = '')
        encoding_output.append(coding[str(idx)])

    string = ''.join([str(item) for item in encoding_output])
    return string


""" A helper function to calculate the space difference between compressed and non compressed data"""


def Total_Gain(coding, hist):
    after_compression = 0
    symbols = coding.keys()
    for symbol in symbols:
        after_compression += (hist[int(symbol)] * len(coding[symbol])) + 1

    return after_compression


def Huffman_Encoding(data, hist):
    symbol_with_probs = Calculate_Probability(data)
    symbols = symbol_with_probs.keys()
    probabilities = symbol_with_probs.values()

    nodes = []

    # converting symbols and probabilities into huffman tree nodes
    for symbol in symbols:
        nodes.append(Node(symbol_with_probs.get(symbol), symbol))

    while len(nodes) > 1:
        # sort all the nodes in ascending order based on their probability
        nodes = sorted(nodes, key=lambda x: x.prob)
        # for node in nodes:
        #      print(node.symbol, node.prob)

        # pick 2 smallest nodes
        right = nodes[0]
        left = nodes[1]

        left.code = 0
        right.code = 1

        # combine the 2 smallest nodes to create new node
        newNode = Node(left.prob + right.prob, left.symbol + right.symbol, left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    huffman_encoding = Calculate_Codes(nodes[0])

    return Total_Gain(huffman_encoding, hist)


#-------------------------------------------------------------------------
def fp32_to_binary_sequence(data, centers, edges, dict):
    index = np.searchsorted(edges, data)
    out_str = itemgetter(*centers[index])(dict)
    return ''.join(out_str)

def LZ78(message):
    tree_dict, m_len, i = {}, len(message), 0
    parse_length = 0
    count = 0
    last = False
    while i < m_len:
        # case I
        if message[i] not in tree_dict.keys():
            tree_dict[message[i]] = len(tree_dict) + 1
            parse_length += 1
            i += 1
        # case III
        elif i == m_len - 1:
            last = True
            i += 1
        else:
            for j in range(i + 1, m_len):
                # case II
                if message[i:j + 1] not in tree_dict.keys():
                    if j-i >15:
                        count += 1
                    parse_length += 1
                    tree_dict[message[i:j + 1]] = len(tree_dict) + 1
                    i = j + 1
                    break
                # case III
                elif j == m_len - 1:
                    last = True
                    i = j + 1
    parse_bits = (int(np.log2(parse_length)) + 1)
    if last:
        return (parse_bits + 1)*(parse_length+1)
    else:
        return (parse_bits + 1)*(parse_length)




# -------------------------------------------------------------------------



def fp8_143_bin_edges(exponent_bias=10):
    bin_centers = np.zeros(239)
    fp8_binary_dict = {}
    fp8_binary_sequence = np.zeros(239, dtype='U8')
    binary_fraction = np.array([2 ** -1, 2 ** -2, 2 ** -3])
    idx = 0
    for s in range(2):
        for e in range(15):
            for f in range(8):
                if e != 0:
                    exponent = int(format(e, 'b').zfill(4), 2) - exponent_bias
                    fraction = np.sum((np.array(list(format(f, 'b').zfill(3)), dtype=int) * binary_fraction)) + 1
                    bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                    fp8_binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(4) + format(f, 'b').zfill(3)
                    idx += 1
                else:
                    if f != 0:
                        exponent = 1-exponent_bias
                        fraction = np.sum((np.array(list(format(f, 'b').zfill(3)), dtype=int) * binary_fraction))
                        bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                        fp8_binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(4) + format(f,'b').zfill(3)
                        idx += 1
                    else:
                        if s == 0:
                            bin_centers[idx] = 0
                            fp8_binary_dict["0.0"] = "00000000"
                            idx += 1
                        else:
                            pass
    bin_centers = np.sort(bin_centers)
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5
    return bin_centers, bin_edges, fp8_binary_dict
def fp8_152_bin_edges(exponent_bias=15):
    bin_centers = np.zeros(247,dtype=np.float32)
    fp8_binary_dict = {}
    fp8_binary_sequence = np.zeros(247, dtype='U8')
    binary_fraction = np.array([2 ** -1, 2 ** -2],dtype=np.float32)
    idx = 0
    for s in range(2):
        for e in range(31):
            for f in range(4):
                if e != 0:
                    exponent = e - exponent_bias
                    fraction = np.sum((np.array(list(format(f, 'b').zfill(2)), dtype=int) * binary_fraction)) + 1
                    bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                    fp8_binary_dict[bin_centers[idx]] = str(s) + format(e, 'b').zfill(5) + format(f, 'b').zfill(2)
                    idx += 1
                else:
                    if f != 0:
                        exponent = 1-exponent_bias
                        fraction = np.sum((np.array(list(format(f, 'b').zfill(2)), dtype=int) * binary_fraction))
                        bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                        fp8_binary_dict[bin_centers[idx]] = str(s) + format(e, 'b').zfill(5) + format(f,'b').zfill(2)
                        idx += 1
                    else:
                        if s == 0:
                            bin_centers[idx] = 0
                            fp8_binary_dict[0.0] = "00000000"
                            idx += 1
                        else:
                            pass
    bin_centers = np.sort(bin_centers)
    print(bin_centers)
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5
    return bin_centers, bin_edges, fp8_binary_dict


fp8_bin_centers,fp8_bin_edges,fp8_dict = fp8_152_bin_edges()

total_bit = 4096*8*15
distribution1 = stats.gennorm
distribution2 = stats.norm
epoch_num = 100

af_HF_g = np.zeros(epoch_num)
af_HF_n = np.zeros(epoch_num)
af_LZ_bits = np.zeros(epoch_num)

for path in range(15):
    filepath = 'C:\\Zhong-Jing\\ResNet50V2\\'+str(path+1)+'\\'
    for m in range(epoch_num):
        print("-----", m + 1, "-----")
        epoch = shelve.open(filepath + 'Gradient_epoch' + str(m) + '.out', flag='r')
        data = epoch['data']['avr_grad'][4].flatten()

        hist, bin = np.histogram(np.abs(data), bins=fp8_bin_edges[122:])

        codes = dict()
        param_g = distribution1.fit(data)
        param_n = distribution2.fit(data)
        CDF = distribution1.cdf(fp8_bin_edges[122:], param_g[0], loc=param_g[1], scale=param_g[2])
        PMF = CDF[1:] - CDF[:-1]
        bits = Huffman_Encoding(PMF,hist)
        af_HF_g[m] += bits

        # ----------------------------------------------------------------
        codes = dict()
        CDF = distribution2.cdf(fp8_bin_edges[122:], loc=param_n[0], scale=param_n[1])
        PMF = CDF[1:] - CDF[:-1]
        bits = Huffman_Encoding(PMF, hist)
        af_HF_n[m] += bits

        binary_data = fp32_to_binary_sequence(data, fp8_bin_centers, fp8_bin_edges, fp8_dict)
        af_LZ_bits[m] += LZ78(binary_data)

plt.figure()
plt.plot(af_LZ_bits / total_bit)
plt.plot(af_HF_g / total_bit)
plt.plot(af_HF_n / total_bit)
plt.legend(['LZ', 'HF(GG)', 'HF(G)'])
plt.show()

