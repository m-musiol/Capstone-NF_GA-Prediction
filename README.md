# What am I predicting?

We are predicting the natural log of the sum of all transactions per user in test_v2.csv for December 1st, 2018 to January 31st, 2019. 
The dataset does NOT contain data for December 1st 2018 to January 31st 2019. You must identify the unique fullVisitorIds in the provided test_v2.csv and make predictions for them for those unseen months.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>y</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>u</mi>
      <mi>s</mi>
      <mi>e</mi>
      <mi>r</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <munderover>
    <mo>&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>n</mi>
    </mrow>
  </munderover>
  <mi>t</mi>
  <mi>r</mi>
  <mi>a</mi>
  <mi>n</mi>
  <mi>s</mi>
  <mi>a</mi>
  <mi>c</mi>
  <mi>t</mi>
  <mi>i</mi>
  <mi>o</mi>
  <msub>
    <mi>n</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>u</mi>
      <mi>s</mi>
      <mi>e</mi>
      <msub>
        <mi>r</mi>
        <mi>i</mi>
      </msub>
    </mrow>
  </msub>
</math>

y_{user} = \sum_{i=1}^{n} transaction_{user_i}
