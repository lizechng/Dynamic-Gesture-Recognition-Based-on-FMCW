<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
</div>

**News**: We released the report on [ICSIP](https://ieeexplore.ieee.org/document/9339325).

## Dynamic Gesture Recognition Based on FMCW

This project is based on FMCW radar, and completed the data acquisition of four kinds of dynamic gesture signals. Then we use Matlab to generate RDM/xTM images, which is as the input of Multi-Net and TS-FNN. The acc is about 98%.

This works based on **Tensorflow 1.12**.

<div align=center>![demo image](img/arch.png)

### Major images

- **Range-Doppler Map**

  <div align=center>![demo image](img/AWR.png)</div>
  <div align=center>![demo image](img/RDM.png)</div>
  <div align=center>![demo image](img/RDM2.png)</div>
  
- **Multi-Net structure**

  <div align=center>![demo image](img/Multi-Net.png)</div>
  
- **TS-FNN structure**

  <div align=center>![demo image](img/TS-FNN.png)</div>

- **Result**

  <div align=center>![demo image](img/result.png)</div>

## Code explaination

- [x] GeneratexTM: Generate Angle-Time/Range-Time/Doppler-Time Map.
- [x] GenerateRDM: Generate Range-Doppler Map.
- [x] Multi-Net: Nerual network code with ATM/RTM/DTM as the input, just using 2d-cnn.
- [x] TS-FNN: Nerual network code with ATM/RDM as the input, using 3d-cnn and LSTM.
- [x] tools: some python scripts

## License

This project is released under the [Apache 2.0 license](LICENSE).
