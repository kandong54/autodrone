import { Component, OnInit, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { Router } from '@angular/router';
import { ClientService } from '../grpc/client.service';
import { CameraReply } from '../../protos/drone_pb';
import { ClientReadableStream } from 'grpc-web';
import colormap from 'colormap';

@Component({
  selector: 'app-camera',
  templateUrl: './camera.component.html',
  styleUrls: ['./camera.component.sass']
})
export class CameraComponent implements OnInit, AfterViewInit {

  imageWidth: number = 0;
  imageHeight: number = 0;
  depthSize: number = 0;
  depthFactor: number = 0.2;
  imageStream: ClientReadableStream<CameraReply> | null = null;
  interval: number;
  lastTime: number = 0;
  context2D: CanvasRenderingContext2D | null = null;
  colors = colormap({
    colormap: 'jet',
    nshades: 256,
    format: 'rgba',
    alpha: 255
  });

  @ViewChild('myCanvas')
  private myCanvas: ElementRef = {} as ElementRef;

  constructor(private clientService: ClientService,
    private router: Router) {
    // estimated interval
    this.interval = 100;
  }

  ngOnInit(): void {
    this.clientService.connect()
      .then(result => console.log('Connect result:', result))
      .catch((error) => {
        alert('Connection error: ' + error);
        this.router.navigate(['/login']);
      });

  }

  updateFps(): void {
    let thisTime = Date.now();
    let delta = thisTime - this.lastTime;
    this.lastTime = thisTime;
    let k = 0.95;
    this.interval = this.interval * k + delta * (1 - k);
  }

  handleCamera(response: CameraReply): void {
    // fps
    this.updateFps();
    // read
    let imageJpeg = response.getImage_asU8();
    let depthU8 = response.getDepth_asU8();
    let boxes = response.getBoxList();
    let image = new Image();
    // TODO: offscreen render
    // depth
    if (!(this.depthSize)) {
      this.depthSize = Math.sqrt(depthU8.length);
    }
    // jpg
    let blob = new Blob([imageJpeg], { 'type': 'image/jpeg' });
    image.src = URL.createObjectURL(blob);
    image.onload = () => {
      if (this.context2D === null) {
        return;
      }
      let imageDepth = this.context2D.createImageData(this.depthSize, this.depthSize);
      for (let y = 0; y < this.depthSize; y++) {
        for (let x = 0; x < this.depthSize; x++) {
          imageDepth.data[(y * this.depthSize + x) * 4] = this.colors[depthU8[y * this.depthSize + x]][0];
          imageDepth.data[(y * this.depthSize + x) * 4 + 1] = this.colors[depthU8[y * this.depthSize + x]][1];
          imageDepth.data[(y * this.depthSize + x) * 4 + 2] = this.colors[depthU8[y * this.depthSize + x]][2];
          imageDepth.data[(y * this.depthSize + x) * 4 + 3] = 255;
        }
      }
      this.context2D.clearRect(0, 0, this.context2D.canvas.width, this.context2D.canvas.height);
      this.context2D.drawImage(image, 0, 0, this.imageWidth, this.imageHeight, 0, 0, this.context2D.canvas.width, this.context2D.canvas.height);
      this.context2D.fillText('FPS: ' + (1000 / this.interval).toFixed(2), 10, 50);
      this.context2D.lineWidth = 3;
      for (const box of boxes) {
        let x = box.getLeft();
        let y = box.getTop();
        let width = box.getWidth();
        let height = box.getHeight();
        this.context2D.strokeRect(x, y, width, height);
        let confidence = box.getConfidence();
        let depth = box.getDepth();
        this.context2D.fillText(Math.round(confidence * 100) + '% ' + Math.round(depth) + 'cm', x + 5, y + height - 6);
      }
      createImageBitmap(imageDepth).then((bitmap) => {
        if (this.context2D === null) {
          return;
        }
        this.context2D.drawImage(bitmap, 0, 0, this.depthSize, this.depthSize,
          this.context2D.canvas.width * (1 - this.depthFactor), this.context2D.canvas.height * (1 - this.depthFactor),
          this.context2D.canvas.width * this.depthFactor, this.context2D.canvas.height * this.depthFactor);
        this.context2D.lineWidth = 1;
        for (const box of boxes) {
          let x = box.getLeft() * this.depthFactor + this.imageWidth * (1 - this.depthFactor);
          let y = box.getTop() * this.depthFactor + this.imageHeight * (1 - this.depthFactor);
          let width = box.getWidth() * this.depthFactor;
          let height = box.getHeight() * this.depthFactor;
          this.context2D.strokeRect(x, y, width, height);
        }
      })
      URL.revokeObjectURL(image.src);
    }
  }

  resizeCanvas(canvas: HTMLCanvasElement, ratio: number): void {
    // Get the size of remaining screen
    let width = window.innerWidth;
    let height = window.innerHeight - canvas.offsetTop;
    if (width / height > ratio) {
      canvas.style.width = Math.floor(height * ratio) + 'px';
      canvas.style.height = height + 'px';
    } else {
      canvas.style.height = Math.floor(width / ratio) + 'px';
      canvas.style.width = width + 'px';
    }
    // canvas.style.width = '50%';
    // canvas.style.height = '50%';
  }

  async getImageSize(): Promise<void> {
    let imageSize = await this.clientService.getImageSize();
    if (imageSize === null) {
      throw new Error("Failed to get image size!");
    }
    [this.imageWidth, this.imageHeight, this.depthSize] = imageSize;
    this.myCanvas.nativeElement.width = this.imageWidth;
    this.myCanvas.nativeElement.height = this.imageHeight;
    this.resizeCanvas(this.myCanvas.nativeElement, this.imageWidth / this.imageHeight);
    window.addEventListener("resize", () =>
      this.resizeCanvas(this.myCanvas.nativeElement, this.imageWidth / this.imageHeight));
  }

  async ngAfterViewInit(): Promise<void> {
    await this.getImageSize();

    this.context2D = this.myCanvas.nativeElement.getContext('2d');
    if (this.context2D) {
      // Stroke
      this.context2D.strokeStyle = 'red';
      this.context2D.lineWidth = 3;
      // Font
      this.context2D.font = '48px Roboto';
      // Shadow
      this.context2D.shadowColor = "white";
      this.context2D.shadowBlur = 3;
      this.context2D.shadowOffsetX = 1;
      this.context2D.shadowOffsetY = 1;
    }
    this.imageStream = this.clientService.getCamera();
    if (this.imageStream === null) {
      alert('Failed to get camera data!');
      return;
    }
    this.imageStream.on('error', (err) => alert('Connection error: ' + err));
    this.imageStream.on('end', () => alert('Connection ends.'));
    this.imageStream.on('data', (response) => this.handleCamera(response));
    this.lastTime = Date.now();
  }
}
