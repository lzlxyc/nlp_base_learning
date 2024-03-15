# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

利用训练好的模型做inference。

Author: pankeyu
Date: 2022/10/26
"""
from rich import print

import torch
from transformers import AutoTokenizer

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/comment_classify_new3/model_best/')
model = torch.load('./checkpoints/comment_classify_new3/model_best/model.pt')
model.to(device).eval()

def test_inference(text1, text2, max_seq_len=128) -> torch.tensor:
    """
    预测函数，输入两句文本，返回这两个文本相似/不相似的概率。

    Args:
        text1 (str): 第一段文本
        text2 (_type_): 第二段文本
        max_seq_len (int, optional): 文本最大长度. Defaults to 128.
    
    Reuturns:
        torch.tensor: 相似/不相似的概率 -> (batch, 2)
    """
    encoded_inputs = tokenizer(
        text=[text1],
        text_pair=[text2],
        truncation=True,
        max_length=max_seq_len,
        return_tensors='pt',
        padding='max_length')
    
    with torch.no_grad():
        model.eval()
        logits = model(input_ids=encoded_inputs['input_ids'].to(device),
                        token_type_ids=encoded_inputs['token_type_ids'].to(device),
                        attention_mask=encoded_inputs['attention_mask'].to(device))
        # print(logits)
        score = logits.cpu().numpy()[0]
        false_score, true_score = score[0], score[1]
        res = int(true_score > false_score)
        return res
        # print("匹配结果：",res)


if __name__ == '__main__':
    senc_data = [
        ('车辆违停：该类数据一般为市民等发布的关于某地域存在车辆违停问题的举报、投诉、曝光等信息','【标题】：这两台车违停造成交通拥堵！2024年2月19日上午9：40起这两台车停在违停抓拍监控拍摄不到的位置，造成交通堵塞#重庆秀山#秀山交巡警#重庆交巡警 #违停 @重庆交巡警【正文】：这两台车违停造成交通拥堵！2024年2月19日上午9：40起这两台车停在违停抓拍监控拍摄不到的位置，造成交通堵塞#重庆秀山#秀山交巡警#重庆交巡警 #违停 @重庆交巡警【封面_OCR】：真香圆饺子馆干 小3NS3 【抽帧_OCR】：真香园饺子信是 11 新日|10:29 HD 78% THD 6'),
        ('车辆违停：该类数据一般为市民等发布的关于某地域存在车辆违停问题的举报、投诉、曝光等信息','【标题】：喜欢乱停到单位门口，电话也不留一个，堵你几个小时在说【正文】：喜欢乱停到单位门口，电话也不留一个，堵你几个小时在说【封面_OCR】：【抽帧_OCR】：【语音转写】：	1'),
        ('车辆违停：该类数据一般为市民等发布的关于某地域存在车辆违停问题的举报、投诉、曝光等信息','【标题】：@南昌红谷滩交警 @江西交警 乱停放堵住了出口【正文】：@南昌红谷滩交警 @江西交警 乱停放堵住了出口【封面_OCR】：赣AB3589 【抽帧_OCR】： 生意 工区城 赣A·B3589 量【语音转写】：	1'),
        ('业主维权：该类数据一般为业主发布的举报、投诉、曝光、维权信息，诉求是维护自己房屋买卖、房屋居住过程中的合法权益','【标题】：喜欢上你的时候 我甚至连你的脸都没看清【正文】：喜欢上你的时候 我甚至连你的脸都没看清【封面_OCR】：Q快影 【抽帧_OCR】：快影【语音转写】：'),
        ('业主维权：该类数据一般为业主发布的举报、投诉、曝光、维权信息，诉求是维护自己房屋买卖、房屋居住过程中的合法权益','【标题】：《深海》让人瞬间泪崩的台词【正文】：-【封面_OCR】：《深海》让人瞬间泪崩的台词 【抽帧_OCR】：风月诗词馆ililil 《深海》让人瞬间泪崩的台词 “心结要是解不开永远不会开心的” 【语音转写】：到最后，静静星于夕阳能留在身上，对不及将故事多跌倒归去去却你我中没想机会。到最后，静静星与夕阳能留在身上来，不及想故事多年。')
    ]

    for idx,(senc1,senc2) in enumerate(senc_data):
        print(idx)
        test_inference(
            senc1,
            senc2,
            max_seq_len=128
        )


