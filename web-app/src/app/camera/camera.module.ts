import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { CameraRoutingModule } from './camera-routing.module';
import { CameraComponent } from './camera.component';


@NgModule({
  declarations: [
    CameraComponent
  ],
  imports: [
    CommonModule,
    CameraRoutingModule
  ]
})
export class CameraModule { }
