import { CommonModule } from "@angular/common";
import {
    Component,
    ElementRef,
    inject,
    Signal,
    viewChild,
} from "@angular/core";
import { ConfigService } from "@services/config.service";

@Component({
    selector: "app-canvas",
    template: `
        <div
            #frame
            class="fixed left-0 top-0 h-screen w-screen cursor-default overflow-scroll">
            <div class="canvas" [ngClass]="{ grid: showGrid() }">
                <ng-content></ng-content>
            </div>
        </div>
    `,
    styles: `
        .canvas {
            position: relative;
            width: 4000px;
            height: 4000px;
            overflow: hidden;
        }

        .canvas.grid {
            background-image: radial-gradient(
                circle,
                hsl(var(--border)) 1.5px,
                transparent 1.5px
            );
            background-size: 20px 20px;
        }
    `,
    imports: [CommonModule],
})
export class CanvasComponent {
    public readonly config: ConfigService = inject(ConfigService);
    public readonly frame: Signal<ElementRef<HTMLDivElement>> =
        viewChild.required("frame");
    protected readonly showGrid: Signal<boolean> = this.config.grid;
}
