#!/usr/bin/env python3

import re

def time_convert(ms):
    ms = int(ms)
    tail = ms % 1000
    s = ms // 1000
    mi = s // 60
    s = s % 60
    h = mi // 60
    mi = mi % 60
    h = "00" if h == 0 else str(h)
    mi = "00" if mi == 0 else str(mi)
    s = "00" if s == 0 else str(s)
    tail = str(tail).zfill(3)
    if len(h) == 1: h = '0' + h
    if len(mi) == 1: mi = '0' + mi
    if len(s) == 1: s = '0' + s
    return "{}:{}:{},{}".format(h, mi, s, tail)

def str2list(text):
    pattern = re.compile(r'[\u4e00-\u9fff]|[\w-]+', re.UNICODE)
    elements = pattern.findall(text)
    return elements

class Text2SRT():
    # 终结性标点（句尾合法结束符）
    TERMINAL_PUNCTUATION_ZH = re.compile(r'[。！？]$', re.UNICODE)
    TERMINAL_PUNCTUATION_EN = re.compile(r'[.!?]$', re.UNICODE)
    
    # 所有常见非终结性标点（需要被“升级”为句号的）
    NON_TERMINAL_PUNCTUATION = re.compile(r'[,:;，、；：]$', re.UNICODE)

    def __init__(self, text, timestamp, offset=0):
        self.token_list = text
        self.timestamp = timestamp
        start, end = timestamp[0][0] - offset, timestamp[-1][1] - offset
        self.start_sec, self.end_sec = start, end
        self.start_time = time_convert(start)
        self.end_time = time_convert(end)

    def _is_chinese_char(self, char):
        """判断字符是否为中文字符"""
        return '\u4e00' <= char <= '\u9fff'

    def _is_chinese_context(self, text):
        """判断一段文本是否属于中文语境（中文字符占优）"""
        chars = [c for c in text if c.isalnum() or self._is_chinese_char(c)]
        if not chars:
            return False
        zh_count = sum(1 for c in chars if self._is_chinese_char(c))
        return zh_count / len(chars) > 0.5

    def text(self):
        if isinstance(self.token_list, str):
            text = self.token_list.strip()
            if not text:
                return ""
            # 已以合法终结标点结尾（！？。!?）→ 保留
            if (self.TERMINAL_PUNCTUATION_ZH.search(text) or 
                self.TERMINAL_PUNCTUATION_EN.search(text)):
                return text
            # 以其他标点结尾（如 , ; : ， ； ： 、）→ 替换为句号
            if self.NON_TERMINAL_PUNCTUATION.search(text):
                is_zh = self._is_chinese_context(text)
                return re.sub(self.NON_TERMINAL_PUNCTUATION, '。' if is_zh else '.', text)
            # 无任何标点结尾 → 添加合适句号
            is_zh = self._is_chinese_context(text)
            return text + ('。' if is_zh else '.')
        else:
            res = ""
            for word in self.token_list:
                if self._is_chinese_char(word):
                    res += word
                else:
                    res += " " + word
            res = res.strip()
            if not res:
                return ""
            # 同上三种情况
            if (self.TERMINAL_PUNCTUATION_ZH.search(res) or 
                self.TERMINAL_PUNCTUATION_EN.search(res)):
                return res
            if self.NON_TERMINAL_PUNCTUATION.search(res):
                is_zh = self._is_chinese_context(res)
                return re.sub(self.NON_TERMINAL_PUNCTUATION, '。' if is_zh else '.', res)
            is_zh = self._is_chinese_context(res)
            return res + ('。' if is_zh else '.')

    def srt(self, acc_ost=0.0):
        return "{} --> {}\n{}\n".format(
            time_convert(self.start_sec + acc_ost * 1000),
            time_convert(self.end_sec + acc_ost * 1000),
            self.text()
        )

    def time(self, acc_ost=0.0):
        return (self.start_sec / 1000 + acc_ost, self.end_sec / 1000 + acc_ost)




def generate_srt(sentence_list):
    srt_total = ''
    for i, sent in enumerate(sentence_list):
        t2s = Text2SRT(sent['text'], sent['timestamp'])
        if 'spk' in sent:
            srt_total += "{}  spk{}\n{}".format(i + 1, sent['spk'], t2s.srt())
        else:
            srt_total += "{}\n{}".format(i + 1, t2s.srt())
    return srt_total

def generate_srt_clip(sentence_list, start, end, begin_index=0, time_acc_ost=0.0):
    # start = int(start * 1000)   # 转成毫秒
    # end = int(end * 1000)
    start = int(round(start * 1000 + 1e-5))
    end = int(round(end * 1000 + 1e-5))
    srt_total = ''
    cc = 1 + begin_index
    subs = []
    for _, sent in enumerate(sentence_list):
        # if isinstance(sent['text'], str):
        #     sent['text'] = str2list(sent['text'])
        if sent['timestamp'][-1][1] <= start:
            continue
        if sent['timestamp'][0][0] >= end:
            break
        # parts in between
        if (sent['timestamp'][-1][1] <= end and sent['timestamp'][0][0] > start) or (sent['timestamp'][-1][1] == end and sent['timestamp'][0][0] == start):
            t2s = Text2SRT(sent['text'], sent['timestamp'], offset=start)
            if 'spk' in sent:
                srt_total += "{}  spk{}\n{}".format(cc, sent['spk'], t2s.srt(time_acc_ost))
            else:
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
            subs.append((t2s.time(time_acc_ost), t2s.text()))
            cc += 1
            continue
        if sent['timestamp'][0][0] <= start:
            if not sent['timestamp'][-1][1] > end:
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        break
                _text = sent['text'][j:]
                _ts = sent['timestamp'][j:]
            else:
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        _start = j
                        break
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > end:
                        _end = j
                        break
                # _text = " ".join(sent['text'][_start:_end])
                _text = sent['text'][_start:_end]
                _ts = sent['timestamp'][_start:_end]
            if _ts and len(_ts) > 0:
                t2s = Text2SRT(_text, _ts, offset=start)
                if 'spk' in sent:
                    srt_total += "{}  spk{}\n{}".format(cc, sent['spk'], t2s.srt(time_acc_ost))
                else:
                    srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append((t2s.time(time_acc_ost), t2s.text()))
                cc += 1
            continue
        if sent['timestamp'][-1][1] > end:
            for j, ts in enumerate(sent['timestamp']):
                if ts[1] > end:
                    break
            _text = sent['text'][:j]
            _ts = sent['timestamp'][:j]
            if _ts and len(_ts) > 0:
                t2s = Text2SRT(_text, _ts, offset=start)
                if 'spk' in sent:
                    srt_total += "{}  spk{}\n{}".format(cc, sent['spk'], t2s.srt(time_acc_ost))
                else:
                    srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append(
                    (t2s.time(time_acc_ost), t2s.text())
                    )
                cc += 1
            continue
    return srt_total, subs, cc



def process_asr_to_sentence_info(rec_result):
    """
    将ASR结果转换为sentence_info格式
    """
    if not rec_result or not hasattr(rec_result, 'text') or not hasattr(rec_result, 'time_stamps'):
        return []
    
    full_text = rec_result.text.strip()
    time_stamps = rec_result.time_stamps
    if not full_text or not time_stamps or len(time_stamps) == 0:
        return []
    
    # 简单的按标点切分逻辑
    sentence_list = []
    start_pos = 0
    punctuation_marks = ('.', '!', '?')
    
    for i, char in enumerate(full_text):
        if char in punctuation_marks:
            sentence_list.append(full_text[start_pos:i+1].strip())
            start_pos = i + 1
    if start_pos < len(full_text):
        tail = full_text[start_pos:].strip()
        if tail: sentence_list.append(tail)
    if not sentence_list:
        return []

    # 构建基础句子块
    sentence_blocks = []
    ts_idx = 0
    total_ts = len(time_stamps)

    for sent in sentence_list:
        # 统计该句中有多少个“词”（排除纯标点）
        # 注意：这里的逻辑假设 text 里的词数和 time_stamps 长度是一致的
        words_in_sent = sent.split()
        if not words_in_sent:
            continue
            
        num_words = len(words_in_sent)
        end_ts_idx = min(ts_idx + num_words, total_ts)
        
        if ts_idx >= total_ts:
            break
            
        current_ts_slice = time_stamps[ts_idx:end_ts_idx]
        
        block = {
            'text': sent,
            'start_ms': round(current_ts_slice[0].start_time * 1000),
            'end_ms': round(current_ts_slice[-1].end_time * 1000),
            'timestamp': [[round(t.start_time * 1000), round(t.end_time * 1000)] for t in current_ts_slice],
            'raw_text': ' '.join([t.text for t in current_ts_slice]),
            'word_count': len(current_ts_slice)
        }
        sentence_blocks.append(block)
        ts_idx = end_ts_idx

    # ===== 4. 贪心合并块 (Merge Blocks) =====
    result = []
    temp_block = sentence_blocks[0]  # 初始化为第一个块
    
    for block in sentence_blocks[1:]:
        current_dur = temp_block['end_ms'] - temp_block['start_ms']
        
        # 合并条件：当前块时长<4秒 且 合并后单词数≤500 且 合并后时长≤60秒
        if (current_dur < 4000 and 
            (temp_block['word_count'] + block['word_count']) <= 500 and 
            (block['end_ms'] - temp_block['start_ms']) <= 60000):
            
            # 执行合并
            temp_block['text'] += " " + block['text']
            temp_block['end_ms'] = block['end_ms']
            temp_block['timestamp'].extend(block['timestamp'])
            temp_block['raw_text'] += " " + block['raw_text']
            temp_block['word_count'] += block['word_count']
        else:
            # 提交当前缓存块（无论是否满足约束）
            result.append({
                'text': temp_block['text'],
                'start': temp_block['start_ms'],
                'end': temp_block['end_ms'],
                'timestamp': temp_block['timestamp'],
                'raw_text': temp_block['raw_text']
            })
            temp_block = block  # 当前块成为新缓存块
    
    # 提交最后剩余的块
    result.append({
        'text': temp_block['text'],
        'start': temp_block['start_ms'],
        'end': temp_block['end_ms'],
        'timestamp': temp_block['timestamp'],
        'raw_text': temp_block['raw_text']
    })

    return result, full_text
