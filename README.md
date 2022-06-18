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
    <td width="24%">入力</td>
    <td width="24%">簡略化(50%)</td>
    <td width="24%">簡略化(20%)</td>
    <td width="24%">簡略化(1%)</td>
  </tr>
  <tr>
    <td width="24%"><img src="docs/original.png" width="100%"/></td>
    <td width="24%"><img src="docs/simp_v1.png" width="100%"/></td>
    <td width="24%"><img src="docs/simp_v2.png" width="100%"/></td>
    <td width="24%"><img src="docs/simp_v4.png" width="100%"/></td>
  </tr>
  
  <tr>
    <td width="24%">14762 vertices</td>
    <td width="24%">7381 vertices</td>
    <td width="24%">2952 vertices</td>
    <td width="24%">147 vertices</td>
  </tr>
  <tr>
    <td width="24%">29520 faces</td>
    <td width="24%">14758 faces</td>
    <td width="24%">5900 faces</td>
    <td width="24%">290 faces</td>
  </tr>
</table>

本スクリプトは改良途中。
非多様体を生じるエッジ縮約や、境界エッジの縮約は行わない。
エッジ角度を考慮していないため、自己交差や面のフリップが生じうる。