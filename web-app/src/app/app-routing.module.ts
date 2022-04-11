import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { LoginComponent } from './login/login.component';

const routes: Routes = [
  { path: '', redirectTo: '/login', pathMatch: 'full' },
  { path: 'login', component: LoginComponent },
  { path: '404.html', loadChildren: () => import('./page-not-found/page-not-found.module').then(m => m.PageNotFoundModule) },
  { path: 'camera', loadChildren: () => import('./camera/camera.module').then(m => m.CameraModule) },
  { path: '**', redirectTo: '/404.html' },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
