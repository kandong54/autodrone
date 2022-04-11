import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { ClientService } from '../grpc/client.service';

@Component({
  selector: 'app-camera',
  templateUrl: './camera.component.html',
  styleUrls: ['./camera.component.sass']
})
export class CameraComponent implements OnInit {

  @ViewChild('myCanvas')
  private myCanvas: ElementRef = {} as ElementRef;

  constructor(private clientService: ClientService) {
  }

  ngOnInit(): void {
    this.clientService.connect()
      .then(result => {
        console.log('Connect result:', result);
      })
      .catch((error) => alert(error));
  }

}
