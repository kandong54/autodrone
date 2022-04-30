import { Component, OnInit, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { Router } from '@angular/router';
import { ClientService } from '../grpc/client.service';
import { CameraReply } from '../../protos/drone_pb';
import { ClientReadableStream } from 'grpc-web';

@Component({
  selector: 'app-camera',
  templateUrl: './camera.component.html',
  styleUrls: ['./camera.component.sass']
})
export class CameraComponent implements OnInit, AfterViewInit {

  cameraWidth: number = 0;
  cameraHeight: number = 0;
  imageWidth: number = 0;
  imageHeight: number = 0;
  imageStream: ClientReadableStream<CameraReply> | null = null;
  interval: number;
  lastTime: number = 0;
  context2D: CanvasRenderingContext2D | null = null;

  @ViewChild('myCanvas')
  private myCanvas: ElementRef = {} as ElementRef;

  constructor(private clientService: ClientService,
    private router: Router) {
    // estimated interval
    this.interval = 230;
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
    let k = 0.9;
    this.interval = this.interval * k + delta * (1 - k);
  }

  handleCamera(response: CameraReply): void {
    // fps
    this.updateFps();
    // read
    let imageRGB = response.getImage_asU8();
    let boxes = response.getBoxList();
    let image = new Image();
    // TODO: offscreen render
    // jpg
    let blob = new Blob([imageRGB], { 'type': 'image/jpeg' });
    image.src = URL.createObjectURL(blob);
    image.onload = () => {
      if (this.context2D === null) {
        return;
      }
      this.context2D.drawImage(image, 0, 0, this.imageWidth, this.imageHeight, 0, 0, this.context2D.canvas.width, this.context2D.canvas.height);
      this.context2D.fillText('FPS: ' + (1000 / this.interval).toFixed(2), 10, 50);
      for (const box of boxes) {
        let x = box.getLeft();
        let y = box.getTop();
        let width = box.getWidth();
        let height = box.getHeight();
        this.context2D.strokeRect(x, y, width, height);
        let confidence = box.getConfidence();
        this.context2D.fillText(Math.round(confidence * 100) + '%', x + 5, y + height - 6);
      }
      URL.revokeObjectURL(image.src);
    }
  }

  resizeCanvas(canvas: HTMLCanvasElement, ratio: number): void {
    // Get the size of remaining screen
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    let width = canvas.offsetWidth;
    let height = canvas.offsetHeight;
    if (width / height > ratio) {
      canvas.style.width = Math.floor(height * ratio) + 'px';
    } else {
      canvas.style.height = Math.floor(width / ratio) + 'px';
    }
    // canvas.style.width = '50%';
    // canvas.style.height = '50%';
  }

  async getImageSize(): Promise<void> {
    let imageSize = await this.clientService.getImageSize();
    if (imageSize === null) {
      throw new Error("Failed to get image size!");
    }
    [this.imageWidth, this.imageHeight, this.cameraWidth, this.cameraHeight] = imageSize;
    this.myCanvas.nativeElement.width = this.cameraWidth;
    this.myCanvas.nativeElement.height = this.cameraHeight;
    this.resizeCanvas(this.myCanvas.nativeElement, this.cameraWidth / this.cameraHeight);
    window.addEventListener("resize", () =>
      this.resizeCanvas(this.myCanvas.nativeElement, this.cameraWidth / this.cameraHeight));
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
