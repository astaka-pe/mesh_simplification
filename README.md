# メッシュ簡略化
"Surface Simplification Using Quadric Error Metrics, 1997" [[Paper]](http://www.cs.cmu.edu/~garland/Papers/quadrics.pdf) を実装。

## ライブラリ
```
numpy
torch
```

## デモ

```
python simplification.py
```

<table>
  <tr>
    <td width="30%">入力</td>
    <td width="30%">簡略化(50%)</td>
    <td width="30%">簡略化(20%)</td>
  </tr>
  <tr>
    <td width="30%"><img src="docs/original.png" width="100%"/></td>
    <td width="30%"><img src="docs/simp_v1.png" width="100%"/></td>
    <td width="30%"><img src="docs/simp_v2.png" width="100%"/></td>
  </tr>
  <tr>
    <td width="30%">14762 vertices</td>
    <td width="30%">7381 vertices</td>
    <td width="30%">2952 vertices</td>
  </tr>
  <tr>
    <td width="30%">29520 faces</td>
    <td width="30%">14758 faces</td>
    <td width="30%">5900 faces</td>
  </tr>
</table>

本スクリプトは改良途中。
非多様体がエッジ縮約ができないため、現状の実装では一定数までのメッシュ簡略化に制約される。