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

  cameraWidth: number;
  cameraHeight: number;
  imageWidth: number;
  imageHeight: number;
  imageStream: ClientReadableStream<CameraReply> | null = null;

  @ViewChild('myCanvas')
  private myCanvas: ElementRef = {} as ElementRef;

  constructor(private clientService: ClientService,
    private router: Router) {
    // TODO: get width and height
    this.cameraWidth = 960;
    this.cameraHeight = 720;
    this.imageWidth = 640;
    this.imageHeight = 640;
  }

  ngOnInit(): void {
    this.clientService.connect()
      .then(result => {
        console.log('Connect result:', result);
      })
      .catch((error) => {
        alert('Connection error: ' + error);
        this.router.navigate(['/login']);
      });

  }


  ngAfterViewInit(): void {
    this.myCanvas.nativeElement.width = this.imageWidth;
    this.myCanvas.nativeElement.height = this.imageHeight;
    this.myCanvas.nativeElement.style.aspectRatio = this.cameraWidth / this.cameraHeight;
    // this.myCanvas.nativeElement.style.width = '100%';
    // this.myCanvas.nativeElement.style.height = 100 * this.cameraHeight / this.cameraWidth + '%';
    var ctx = this.myCanvas.nativeElement.getContext('2d');
    ctx.strokeStyle = 'red';
    ctx.font = '48px Roboto';
    this.clientService.connect()
      .then((result) => {
        if (result === true) {
          this.imageStream = this.clientService.getCamera();
          if (this.imageStream === null) {
            alert('Failed to get camera data!');
            return;
          }
          this.imageStream.on('error', (err) => alert('Connection error: ' + err));
          this.imageStream.on('end', () => alert('Connection ends.'));
          // this.imageStream.on('status', (status) => alert('Connection status: ' + status.details));
          // let imageRGBA = new Uint8ClampedArray(this.imageWidth * this.imageHeight * 4);
          // imageRGBA.fill(255);
          let image = new Image();
          this.imageStream.on('data', (response) => {
            let imageRGB = response.getImage_asU8();
            let boxes = response.getBoxList();
            // TODO: offscreen render
            // raw data
            // for (let iRGBA = 0, iRGB = 0; iRGBA < imageRGBA.length; iRGBA += 4, iRGB += 3) {
            //   imageRGBA[iRGBA] = imageRGB[iRGB];
            //   imageRGBA[iRGBA + 1] = imageRGB[iRGB + 1];
            //   imageRGBA[iRGBA + 2] = imageRGB[iRGB + 2];
            // }
            // let image = new ImageData(imageRGBA, this.imageWidth, this.imageHeight);
            // this.ctx.putImageData(image, 0, 0);
            // jpg
            let blob = new Blob([imageRGB], { 'type': 'image/jpeg' });
            URL.revokeObjectURL(image.src);
            image.src = URL.createObjectURL(blob);
            image.onload = () => {
              ctx.drawImage(image, 0, 0);
              for (const box of boxes) {
                let xCenter = box.getXCenter() * this.imageWidth;
                let yCenter = box.getYCenter() * this.imageHeight;
                let width = box.getWidth() * this.imageWidth;
                let height = box.getHeight() * this.imageHeight;
                let x = xCenter - width / 2;
                let y = yCenter - height / 2;
                ctx.strokeRect(x, y, width, height);
                let confidence = box.getConfidence();
                ctx.fillText(Math.round(confidence * 100) + '%', x, y);
              }
            }
          });
        }
      });
  }
}
